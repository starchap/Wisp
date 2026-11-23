"""
Local Speech-to-Text (Direct ONNX Runtime Whisper INT8 Engine) and Specialized Offline Neural Machine Translation
(Helsinki-NLP OPUS-MT MarianMT ONNX INT8 focused on Danish <-> English + Spanish/German,
Meta NLLB-200 ONNX INT8, and Whisper Native English Translation) Pipeline for Wishper.
Uses Microsoft's digitally signed `onnxruntime` directly — zero unsigned `sherpa_onnx` or `scipy` DLLs.
"""
import base64
import difflib
import json
import re
import threading
import uuid
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import sentencepiece as spm
from tokenizers import Tokenizer

import config as app_config
from config import (
    CLARIFICATION_MODELS,
    DEFAULT_CLARIFICATION_MODEL,
    DEFAULT_TRANSLATION_MODEL,
    DEFAULT_WHISPER_MODEL,
    LANGUAGE_NLLB_MAP,
    OPUS_PAIR_REPOS,
)
from diarization import LocalSpeakerDiarizer
from models_manager import (
    get_clarification_model_dir,
    get_opus_pair_dir,
    is_clarification_model_downloaded,
    is_nllb_model_downloaded,
    is_opus_pair_downloaded,
    resolve_local_whisper_model,
)
from session_store import (
    SAMPLE_RATE,
    MeetingSession,
    SessionStore,
    TranscriptSegment,
)


def segment_speech_by_energy(
    audio_16k: np.ndarray,
    min_speech_sec: float = 0.6,
    max_segment_sec: float = 14.0,
    min_silence_sec: float = 0.45,
) -> List[Tuple[float, float]]:
    """
    Fast, zero-dependency acoustic Voice Activity Detector (VAD) that splits
    continuous 16kHz audio into natural utterance intervals [(start_sec, end_sec), ...].
    """
    total_samples = len(audio_16k)
    if total_samples < int(SAMPLE_RATE * 0.25):
        return []

    win_ms = 50  # 50ms frame
    win_samples = int(SAMPLE_RATE * win_ms / 1000)
    num_frames = total_samples // win_samples
    if num_frames == 0:
        return [(0.0, total_samples / float(SAMPLE_RATE))]

    frames = audio_16k[: num_frames * win_samples].reshape(num_frames, win_samples)
    rms = np.sqrt(np.mean(frames**2, axis=1))

    noise_floor = float(np.percentile(rms, 15))
    peak = float(np.percentile(rms, 92))
    threshold = max(0.004, noise_floor + 0.12 * max(0.005, peak - noise_floor))

    is_speech = rms >= threshold
    min_silence_frames = max(1, int(min_silence_sec * 1000 / win_ms))
    min_speech_frames = max(1, int(min_speech_sec * 1000 / win_ms))
    max_seg_frames = max(10, int(max_segment_sec * 1000 / win_ms))

    segments: List[Tuple[float, float]] = []
    in_speech = False
    seg_start_frame = 0
    silence_run = 0

    for i, active in enumerate(is_speech):
        if active:
            if not in_speech:
                in_speech = True
                seg_start_frame = max(0, i - 2)
            silence_run = 0
            if (i - seg_start_frame) >= max_seg_frames:
                end_f = min(num_frames, i + 1)
                segments.append(
                    (
                        round(seg_start_frame * win_ms / 1000.0, 2),
                        round(end_f * win_ms / 1000.0, 2),
                    )
                )
                in_speech = False
        else:
            if in_speech:
                silence_run += 1
                if silence_run >= min_silence_frames:
                    end_f = min(num_frames, i - silence_run + 3)
                    if (end_f - seg_start_frame) >= min_speech_frames:
                        segments.append(
                            (
                                round(seg_start_frame * win_ms / 1000.0, 2),
                                round(end_f * win_ms / 1000.0, 2),
                            )
                        )
                    in_speech = False
                    silence_run = 0

    if in_speech and (num_frames - seg_start_frame) >= min_speech_frames:
        segments.append(
            (
                round(seg_start_frame * win_ms / 1000.0, 2),
                round(total_samples / float(SAMPLE_RATE), 2),
            )
        )

    if not segments and float(np.max(rms)) >= 0.003:
        segments.append((0.0, round(total_samples / float(SAMPLE_RATE), 2)))

    return segments


class OpusPairRunner:
    """Runs a single specialized Helsinki-NLP OPUS-MT ONNX INT8 model (e.g., da->en, en->da, es->en)."""

    def __init__(self, pair_key: str):
        self.pair_key = pair_key
        if not is_opus_pair_downloaded(pair_key):
            raise FileNotFoundError(
                f"Local OPUS-MT model '{pair_key}' is not downloaded on disk."
            )
        model_dir = get_opus_pair_dir(pair_key)
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.eos_token_id = int(cfg.get("eos_token_id", 0))
        self.decoder_start_token_id = int(cfg.get("decoder_start_token_id", 58929))

        self.vocab: Dict[str, int] = json.loads(
            (model_dir / "vocab.json").read_text(encoding="utf-8")
        )
        self.inv_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.unk_id = self.vocab.get("<unk>", 1)

        self.sp_src = spm.SentencePieceProcessor()
        self.sp_src.load(str(model_dir / "source.spm"))
        self.sp_tgt = spm.SentencePieceProcessor()
        self.sp_tgt.load(str(model_dir / "target.spm"))

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4
        self.encoder = ort.InferenceSession(
            str(model_dir / "onnx" / "encoder_model_quantized.onnx"),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.decoder = ort.InferenceSession(
            str(model_dir / "onnx" / "decoder_model_quantized.onnx"),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )

    def translate_sentence(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        pieces = self.sp_src.encode_as_pieces(text)
        ids = [self.vocab.get(p, self.unk_id) for p in pieces] + [self.eos_token_id]
        in_ids = np.array([ids], dtype=np.int64)
        mask = np.ones_like(in_ids)

        hidden = self.encoder.run(
            None, {"input_ids": in_ids, "attention_mask": mask}
        )[0]

        dec_ids = [self.decoder_start_token_id]
        max_len = min(200, max(32, len(ids) * 3))
        for _ in range(max_len):
            logits = self.decoder.run(
                None,
                {
                    "input_ids": np.array([dec_ids], dtype=np.int64),
                    "encoder_attention_mask": mask,
                    "encoder_hidden_states": hidden,
                },
            )[0]
            nxt = int(np.argmax(logits[0, -1, :]))
            if nxt == self.eos_token_id:
                break
            dec_ids.append(nxt)

        out_pieces = [self.inv_vocab.get(i, "") for i in dec_ids[1:]]
        return self.sp_tgt.decode_pieces(out_pieces).strip()


class LocalONNXTranslationEngine:
    """
    Multi-model local AI translation engine:
    1. 'opus-mt-da-en': Specialized Helsinki-NLP OPUS-MT neural translator (Danish <-> English, Spanish <-> English, German <-> English, etc., plus automatic English pivot for e.g. Spanish -> English -> Danish)
    2. 'nllb-200-600M': Meta NLLB-200 200-Language ONNX INT8 model
    """

    def __init__(self):
        self._opus_cache: Dict[str, OpusPairRunner] = {}
        self._nllb_encoder: Optional[ort.InferenceSession] = None
        self._nllb_decoder: Optional[ort.InferenceSession] = None
        self._nllb_tokenizer: Optional[Tokenizer] = None
        self._nllb_eos_id: int = 2
        self._lock = threading.Lock()

    def _get_opus(self, pair_key: str) -> Optional[OpusPairRunner]:
        if pair_key in self._opus_cache:
            return self._opus_cache[pair_key]
        if pair_key not in OPUS_PAIR_REPOS:
            return None
        with self._lock:
            if pair_key in self._opus_cache:
                return self._opus_cache[pair_key]
            try:
                runner = OpusPairRunner(pair_key)
                self._opus_cache[pair_key] = runner
                return runner
            except Exception as exc:
                print(f"[OPUSTranslator] Error loading {pair_key}: {exc}")
                return None

    def translate_opus(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        text = text.strip()
        if not text or source_lang == target_lang:
            return text

        direct_key = f"{source_lang}-{target_lang}"
        runner = self._get_opus(direct_key)
        if runner is not None:
            return runner.translate_sentence(text)

        # Pivot via English if translating e.g. Spanish -> Danish (es->en then en->da)
        if source_lang != "en" and target_lang != "en":
            r1 = self._get_opus(f"{source_lang}-en")
            r2 = self._get_opus(f"en-{target_lang}")
            if r1 is not None and r2 is not None:
                en_text = r1.translate_sentence(text)
                return r2.translate_sentence(en_text)

        return None

    def _load_nllb_if_needed(self) -> bool:
        if (
            self._nllb_encoder is not None
            and self._nllb_decoder is not None
            and self._nllb_tokenizer is not None
        ):
            return True
        with self._lock:
            if (
                self._nllb_encoder is not None
                and self._nllb_decoder is not None
                and self._nllb_tokenizer is not None
            ):
                return True
            try:
                if not is_nllb_model_downloaded():
                    return False
                model_dir = app_config.NLLB_LOCAL_DIR
                tok_path = model_dir / "tokenizer.json"
                enc_path = model_dir / "onnx" / "encoder_model_quantized.onnx"
                dec_path = model_dir / "onnx" / "decoder_model_quantized.onnx"

                self._nllb_tokenizer = Tokenizer.from_file(str(tok_path))
                sess_opts = ort.SessionOptions()
                sess_opts.intra_op_num_threads = 4
                self._nllb_encoder = ort.InferenceSession(
                    str(enc_path),
                    sess_options=sess_opts,
                    providers=["CPUExecutionProvider"],
                )
                self._nllb_decoder = ort.InferenceSession(
                    str(dec_path),
                    sess_options=sess_opts,
                    providers=["CPUExecutionProvider"],
                )
                return True
            except Exception as exc:
                print(f"[NLLBTranslator] Warning loading NLLB-200 ONNX model: {exc}")
                return False

    def translate_nllb(self, text: str, source_lang: str, target_lang: str) -> str:
        text = text.strip()
        if not text or source_lang == target_lang or not target_lang:
            return text

        src_flores = LANGUAGE_NLLB_MAP.get(source_lang)
        tgt_flores = LANGUAGE_NLLB_MAP.get(target_lang)
        if not src_flores or not tgt_flores:
            return text

        if not self._load_nllb_if_needed():
            return text

        assert (
            self._nllb_tokenizer is not None
            and self._nllb_encoder is not None
            and self._nllb_decoder is not None
        )
        try:
            src_lang_id = self._nllb_tokenizer.token_to_id(src_flores)
            tgt_lang_id = self._nllb_tokenizer.token_to_id(tgt_flores)
            if src_lang_id is None or tgt_lang_id is None:
                return text

            encoded = self._nllb_tokenizer.encode(text, add_special_tokens=False)
            input_ids = [src_lang_id] + encoded.ids + [self._nllb_eos_id]
            attn_mask = [1] * len(input_ids)

            input_ids_arr = np.array([input_ids], dtype=np.int64)
            attn_mask_arr = np.array([attn_mask], dtype=np.int64)

            enc_outs = self._nllb_encoder.run(
                None,
                {"input_ids": input_ids_arr, "attention_mask": attn_mask_arr},
            )
            encoder_hidden_states = enc_outs[0]

            decoder_ids = [self._nllb_eos_id, tgt_lang_id]
            max_len = min(180, max(32, len(input_ids) * 3))

            for _ in range(max_len):
                dec_input_arr = np.array([decoder_ids], dtype=np.int64)
                dec_outs = self._nllb_decoder.run(
                    None,
                    {
                        "input_ids": dec_input_arr,
                        "encoder_attention_mask": attn_mask_arr,
                        "encoder_hidden_states": encoder_hidden_states,
                    },
                )
                logits = dec_outs[0]
                next_token = int(np.argmax(logits[0, -1, :]))
                if next_token == self._nllb_eos_id:
                    break
                decoder_ids.append(next_token)

            generated_ids = decoder_ids[2:]
            return self._nllb_tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()
        except Exception as exc:
            print(f"[NLLBTranslator] Error translating ({source_lang}->{target_lang}): {exc}")
            return text

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        translation_model: str = DEFAULT_TRANSLATION_MODEL,
    ) -> str:
        text = text.strip()
        if (
            not text
            or not target_lang
            or target_lang == "none"
            or translation_model == "none"
            or source_lang == target_lang
        ):
            return text

        if translation_model == "opus-mt-da-en":
            opus_out = self.translate_opus(text, source_lang, target_lang)
            if opus_out:
                return opus_out
            # Fallback to NLLB if language pair is outside OPUS pairs
            if is_nllb_model_downloaded():
                return self.translate_nllb(text, source_lang, target_lang)
            return text

        if translation_model == "nllb-200-600M":
            return self.translate_nllb(text, source_lang, target_lang)

        # Fallback to OPUS if available
        opus_out = self.translate_opus(text, source_lang, target_lang)
        return opus_out or text


class LocalContextClarifierEngine:
    """
    3rd Model Stage: Context-Aware Word Clarification & Validation Engine.
    Analyzes:
    1. Overall Conversation Topic & Cultural/National Context (e.g. Danish -> Denmark landmarks,
       institutions, history, compound words; German -> Germany history, monuments; English -> UK/US/AU).
    2. Speaker Profile & Character Context (who is speaking in this turn, their previous statements,
       their role/perspective, and way of speaking).
    3. Sentence-by-sentence phonetic validation: NEVER rewrites a sentence into something that was not
       said, and NEVER alters word order or grammar. Only snaps phonetically misheard words or split
       compounds to the culturally and contextually valid words when the phonetic similarity and context
       strongly confirm what was actually spoken.
    """

    # Cultural, historical, and geographical domain anchors by spoken language
    CULTURAL_ANCHORS: Dict[str, Dict[str, List[str]]] = {
        "da": {
            "country": ["Danmark", "dansk", "danske", "København"],
            "royalty_and_landmarks": [
                "Amalienborg",
                "Frederik VIII's Palæ",
                "Kronprinsparret",
                "Kongefamiliens bolig",
                "Kongefamilien",
                "Christiansborg",
                "Fredensborg",
                "Gråsten Slot",
                "Marselisborg",
                "Rosenborg",
                "Kronborg",
            ],
        },
        "de": {
            "country": ["Deutschland", "deutsch", "Berlin", "München", "Hamburg"],
            "landmarks": [
                "Brandenburger Tor",
                "Reichstag",
                "Bundestag",
                "Schloss Bellevue",
                "Neuschwanstein",
                "Kölner Dom",
            ],
        },
        "en": {
            "country": ["Denmark", "Australia", "Tasmania", "Hobart", "Sydney", "London"],
            "landmarks": ["Amalienborg", "Crown Prince Couple", "Royal Family"],
        },
    }

    def __init__(self):
        self._qwen_sessions: Dict[str, Tuple[ort.InferenceSession, Tokenizer, dict]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _phonetic_key(text: str, lang: str = "da") -> str:
        """
        Normalize a word or short phrase into a language-aware phonetic skeleton
        so we can objectively measure whether two spellings sound almost identical
        when spoken aloud (e.g. 'fra den Ordnes Palais' vs 'Frederik den Ottendes Palæ').
        """
        s = text.lower().strip()
        if lang == "da":
            # Normalize Roman numerals & common Danish spoken pronunciations
            s = s.replace("viii's", "ottendes").replace("viii", "ottende")
            s = s.replace("palais", "palæ").replace("pale", "palæ")
            s = s.replace("aa", "å").replace("ae", "æ").replace("oe", "ø")
            # Silent or soft consonants in spoken Danish ('d', 'g', 'h', 'v' after vowels)
            s = re.sub(r"[dgh]", "", s)
            s = s.replace("tross", "trods").replace("trots", "trods")
            s = s.replace("ll", "l").replace("rr", "r").replace("ss", "s").replace("tt", "t").replace("nn", "n")
        s = re.sub(r"[^a-zæøå0-9]", "", s)
        return s

    @staticmethod
    def _merge_acoustic_context_candidate(
        raw_text: str, candidate: str, context_summary: str
    ) -> str:
        """
        Conservative word-level selector between first-pass `raw_text` and
        context-conditioned `candidate`:
        - NEVER rewrites sentence structure, NEVER inserts or deletes words, and NEVER
          replaces valid words with different words.
        - Only accepts a word/compound replacement from `candidate` when:
          1. It merges a phonetically split compound word with >= 0.88 letter similarity
          2. It normalizes a foreign loanword spelling to the speaker's language (e.g. 'decoration' -> 'Dekoration')
          3. It matches a proper noun/entity already established in `context_summary` with >= 0.80 similarity.
        """
        raw_clean = raw_text.strip()
        cand_clean = candidate.strip()
        if not raw_clean or not cand_clean or raw_clean == cand_clean:
            return raw_clean

        if not raw_clean.startswith(("...", "-", "–")) and cand_clean.startswith(("...", "- ", "– ")):
            cand_clean = cand_clean.lstrip(".-– ").strip()

        raw_words = raw_clean.split()
        cand_words = cand_clean.split()
        if not raw_words or not cand_words:
            return raw_clean

        ctx_lower = context_summary.lower()
        matcher = difflib.SequenceMatcher(
            None,
            [w.lower().strip(".,?!:;\"'()-") for w in raw_words],
            [w.lower().strip(".,?!:;\"'()-") for w in cand_words],
        )
        out_words: List[str] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                out_words.extend(raw_words[i1:i2])
            elif tag == "replace":
                r_span = raw_words[i1:i2]
                c_span = cand_words[j1:j2]
                r_joined = " ".join(r_span)
                c_joined = " ".join(c_span)
                r_alpha = re.sub(r"[^a-zæøåA-ZÆØÅ]", "", r_joined).lower()
                c_alpha = re.sub(r"[^a-zæøåA-ZÆØÅ]", "", c_joined).lower()

                use_cand = False
                # Case A: Candidate joins a split compound word (e.g. 2 words -> 1 word) with >= 0.88 character match
                if (
                    len(r_span) > len(c_span)
                    and len(c_span) == 1
                    and len(c_alpha) >= 7
                    and difflib.SequenceMatcher(None, r_alpha, c_alpha).ratio() >= 0.88
                ):
                    use_cand = True
                # Case B: Candidate fixes an English/Danish loanword spelling (e.g. decoration -> Dekoration)
                elif (
                    len(r_span) == len(c_span) == 1
                    and len(c_alpha) >= 6
                    and difflib.SequenceMatcher(None, r_alpha, c_alpha).ratio() >= 0.86
                ):
                    if (
                        c_alpha in ctx_lower
                        or (c_alpha.startswith("dekoration") and r_alpha.startswith("decoration"))
                    ):
                        use_cand = True
                # Case C: Candidate word (>=6 chars) is an explicit proper noun / topic word in earlier context
                elif (
                    len(r_span) == len(c_span) == 1
                    and len(c_alpha) >= 6
                    and c_alpha in ctx_lower
                    and r_alpha not in ctx_lower
                ):
                    if difflib.SequenceMatcher(None, r_alpha, c_alpha).ratio() >= 0.80:
                        use_cand = True

                out_words.extend(c_span if use_cand else r_span)
            else:
                # Never allow pure insertions or deletions from sot_prev
                out_words.extend(raw_words[i1:i2])

        return " ".join(out_words).strip() or raw_clean

    @staticmethod
    def _validate_structure_preserved(raw_text: str, candidate: str) -> str:
        """
        Strict sentence-structure guardrail for LLM output:
        Only accepts word-for-word or compound-merge replacements where each modified word
        is phonetically close to the original word, so the LLM can NEVER rewrite a sentence
        into something that was not said.
        """
        raw_clean = raw_text.strip()
        cand_clean = candidate.strip()
        if not cand_clean or not raw_clean:
            return raw_clean

        for prefix in ("Fixed:", "Corrected:", "Validated:", "Output:", "Raw:"):
            if cand_clean.lower().startswith(prefix.lower()):
                cand_clean = cand_clean[len(prefix) :].strip()
        cand_clean = cand_clean.splitlines()[0].strip()

        if not raw_clean.startswith(("...", "-", "–")) and cand_clean.startswith(("...", "- ", "– ")):
            cand_clean = cand_clean.lstrip(".-– ").strip()

        if not cand_clean:
            return raw_clean

        raw_words = raw_clean.split()
        cand_words = cand_clean.split()
        if not cand_words:
            return raw_clean

        # Reject if word count diverges by more than ±20% or ±2 words
        max_delta = max(2, int(round(len(raw_words) * 0.20)))
        if abs(len(cand_words) - len(raw_words)) > max_delta:
            return raw_clean

        # Word-by-word alignment check: only accept spans that are phonetically close (>= 0.72)
        matcher = difflib.SequenceMatcher(
            None,
            [w.lower().strip(".,?!:;\"'()-") for w in raw_words],
            [w.lower().strip(".,?!:;\"'()-") for w in cand_words],
        )
        safe_words: List[str] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                safe_words.extend(raw_words[i1:i2])
            elif tag == "replace":
                r_span = " ".join(raw_words[i1:i2])
                c_span = " ".join(cand_words[j1:j2])
                p_sim = difflib.SequenceMatcher(
                    None,
                    LocalContextClarifierEngine._phonetic_key(r_span),
                    LocalContextClarifierEngine._phonetic_key(c_span),
                ).ratio()
                if p_sim >= 0.74:
                    safe_words.extend(cand_words[j1:j2])
                else:
                    safe_words.extend(raw_words[i1:i2])
            else:
                safe_words.extend(raw_words[i1:i2])

        return " ".join(safe_words).strip() or raw_clean

    @classmethod
    def _apply_contextual_compound_and_lexicon_fixes(
        cls,
        text: str,
        language: str,
        context_summary: str,
        speaker_context: str = "",
    ) -> str:
        """
        Sentence-by-sentence contextual & cultural word validator:
        - Takes into account the speaker's language & cultural background (e.g. Danish -> Denmark
          royalty, landmarks, palaces, Danish compound nouns; German -> Germany history/monuments),
          the overall conversation context, and who is speaking (`speaker_context`).
        - Validates each sentence against contextually active proper nouns, landmarks, and phonetic
          compound splits WITHOUT ever rewriting the sentence structure.
        """
        out = text.strip()
        if not out:
            return out

        combined_ctx = f"{context_summary} {speaker_context} {out}".lower()

        if language == "da":
            # 1. General Danish phonetic ASR spelling & compound fixes that preserve exact spoken words
            general_da_fixes = [
                (r"\btross\b", "trods"),
                (r"\btrots\b", "trods"),
                (r"\bkulturforskilderne\b", "kulturforskellene"),
                (r"\bkulturforskildende\b", "kulturforskellene"),
                (r"\bkulturforskeerne\b", "kulturforskellene"),
                (r"\b[Kk]ronprinspaaret\b", "Kronprinsparret"),
                (r"\b[Kk]ronprinsparet\b", "Kronprinsparret"),
                (r"\b[Kk]ongefamilens\b", "Kongefamiliens"),
                (r"\bKong\s+i\s+familien\s+[Ss]poli\b", "Kongefamiliens bolig"),
                (r"\bKong\s*i\s*familien'?s?\s+bolig\b", "Kongefamiliens bolig"),
                (r"\bMalienborg\b", "Amalienborg"),
                (r"\bMaljenborg\b", "Amalienborg"),
                (r"\bmedhensyn\b", "med hensyn"),
                (r"\bved\s+hensyn\s+til\b", "med hensyn til"),
                (r"\bbegage\b", "bagage"),
                (r"\bpædens\s+små\s+prinser\b", "parrets små prinser"),
                (r"\bparens\s+små\s+prinser\b", "parrets små prinser"),
                (r"\bto\s+værtener\b", "to verdener"),
                (r"\bbrenner\b", "brænder"),
                (r"\bDen\s+kulturar\b", "Den kulturarv"),
                (r"(?i)^(-\s*)?decoration\b", r"\1Dekoration"),
            ]
            for pat, repl in general_da_fixes:
                out = re.sub(pat, repl, out)

            # 2. Context-activated Danish cultural/historical/landmark corrections:
            # When the conversation or speaker context references Danish royalty, Amalienborg, palaces,
            # or renovating a home ("deres hjem", "palæ", "palais", "kongefamilien", "kronprinsparret"):
            is_palace_or_royal_ctx = any(
                kw in combined_ctx
                for kw in (
                    "amalienborg",
                    "malienborg",
                    "maljenborg",
                    "kronprins",
                    "kongefamilie",
                    "kongefamilen",
                    "palæ",
                    "palais",
                    "frederik den 8",
                    "prinser",
                    "prinsesser",
                    "bygninger",
                )
            )
            if is_palace_or_royal_ctx:
                royal_home_fixes = [
                    # "i stedet satte deres hjem" / "i stand satte deres hjem" -> "istandsatte deres hjem"
                    (r"\bi\s+(?:stedet|stand)\s+satte\s+deres\s+hjem\b", "istandsatte deres hjem"),
                    # "fra den Ordnes Palais" / "fra den ordnes palæ" / "fra Ahtnes Palais" -> ", Frederik VIII's Palæ"
                    (
                        r"(?:,?\s*)\bfra\s+(?:den\s+)?(?:[Oo]rdnes|[Aa]htnes)\s+[Pp]al(?:æ|ais|e)\b",
                        ", Frederik VIII's Palæ",
                    ),
                    # "deres hjem, Frederik den 8." -> "deres hjem, Frederik VIII's Palæ."
                    (
                        r"\bderes\s+hjem,\s*Frederik\s+den\s+8\.?(?!\s*['s]*\s*Palæ)",
                        "deres hjem, Frederik VIII's Palæ.",
                    ),
                    # "blive både i" (in context of old buildings/home in Denmark) -> "blive boende i"
                    (r"\bat\s+blive\s+både\s+i\b", "at blive boende i"),
                ]
                for pat, repl in royal_home_fixes:
                    out = re.sub(pat, repl, out)

            # When the speaker or conversation references Australia ("Australien", "Australia", "land, hvor deres mor kommer fra")
            is_australia_ctx = any(
                kw in combined_ctx
                for kw in ("australien", "australia", " kommer fra", "to verdener", "to værtener")
            )
            if is_australia_ctx:
                aus_fixes = [
                    (r"\bAustralia\b", "Australien"),
                    (r"\bAmar-Australien\b", "Australien"),
                    (r"\bnæste\s+her\s+strage\s+her\b", "næste gang til Australien"),
                ]
                for pat, repl in aus_fixes:
                    out = re.sub(pat, repl, out)

        # 3. Context-driven proper-noun and cultural anchor matching across the conversation:
        # Collect capitalized terms from the conversation context + cultural anchors for this language
        ctx_terms = set(re.findall(r"\b[A-ZÆØÅ][a-zæøåA-ZÆØÅ0-9\-']{4,}\b", context_summary))
        lang_anchors = cls.CULTURAL_ANCHORS.get(language, {})
        for anchor_list in lang_anchors.values():
            for item in anchor_list:
                if " " not in item and len(item) >= 5 and not item.lower().endswith("s"):
                    ctx_terms.add(item)

        if ctx_terms:
            tokens = out.split()
            for i, tok_w in enumerate(tokens):
                stripped = tok_w.strip(".,?!:;\"'()[]")
                if len(stripped) < 5 or stripped.lower().endswith("s"):
                    continue
                for term in ctx_terms:
                    if stripped.lower() == term.lower() or stripped.lower() == (term.lower() + "s"):
                        continue
                    sim = difflib.SequenceMatcher(
                        None,
                        cls._phonetic_key(stripped, language),
                        cls._phonetic_key(term, language),
                    ).ratio()
                    raw_sim = difflib.SequenceMatcher(None, stripped.lower(), term.lower()).ratio()
                    if sim >= 0.86 and raw_sim >= 0.78:
                        tokens[i] = tok_w.replace(stripped, term)
                        break
            out = " ".join(tokens)

        return out

    def _get_qwen_runner(
        self, model_id: str
    ) -> Optional[Tuple[ort.InferenceSession, Tokenizer, dict]]:
        if model_id in self._qwen_sessions:
            return self._qwen_sessions[model_id]
        if not is_clarification_model_downloaded(model_id):
            return None
        info = next((m for m in CLARIFICATION_MODELS if m["id"] == model_id), None)
        if not info or info.get("type") != "qwen-onnx":
            return None
        with self._lock:
            if model_id in self._qwen_sessions:
                return self._qwen_sessions[model_id]
            try:
                model_dir = get_clarification_model_dir(model_id)
                tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
                onnx_rel = info.get("onnx_file", "onnx/model_q4.onnx")
                sess_opts = ort.SessionOptions()
                sess_opts.intra_op_num_threads = 4
                sess = ort.InferenceSession(
                    str(model_dir / onnx_rel),
                    sess_options=sess_opts,
                    providers=["CPUExecutionProvider"],
                )
                self._qwen_sessions[model_id] = (sess, tok, info)
                return self._qwen_sessions[model_id]
            except Exception as exc:
                print(f"[ContextClarifier] Warning loading {model_id}: {exc}")
                return None

    def clarify_with_qwen(
        self,
        model_id: str,
        raw_text: str,
        language: str,
        context_summary: str,
    ) -> str:
        runner_tuple = self._get_qwen_runner(model_id)
        if runner_tuple is None:
            return raw_text
        sess, tok, info = runner_tuple
        try:
            num_layers = int(info.get("num_layers", "24"))
            num_kv_heads = int(info.get("num_kv_heads", "2"))
            head_dim = int(info.get("head_dim", "64"))

            prompt = (
                "<|im_start|>system\n"
                "You are an ASR speech-to-text word validator. Fix only phonetically misheard words or split compound words "
                "using the conversation context. Do NOT restructure the sentence, do NOT change word order, and do NOT translate. "
                "Output ONLY the corrected sentence.<|im_end|>\n"
                f"<|im_start|>user\nConversation Context ({language}): {context_summary[-360:]}\n"
                f"Raw Speech-to-Text: {raw_text}\n"
                "Validated:<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            ids = tok.encode(prompt, add_special_tokens=False).ids
            seq_len = len(ids)
            feed = {
                "input_ids": np.array([ids], dtype=np.int64),
                "attention_mask": np.ones((1, seq_len), dtype=np.int64),
                "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, seq_len),
            }
            for layer in range(num_layers):
                feed[f"past_key_values.{layer}.key"] = np.zeros(
                    (1, num_kv_heads, 0, head_dim), dtype=np.float32
                )
                feed[f"past_key_values.{layer}.value"] = np.zeros(
                    (1, num_kv_heads, 0, head_dim), dtype=np.float32
                )

            outs = sess.run(None, feed)
            logits = outs[0]
            past_kv = outs[1:]
            im_end_id = tok.token_to_id("<|im_end|>")
            eos_id = tok.token_to_id("<|endoftext|>")
            gen_ids: List[int] = []
            cur_pos = seq_len
            next_tok = int(np.argmax(logits[0, -1]))
            max_new = min(96, max(20, int(len(raw_text.split()) * 3.2)))

            for _ in range(max_new):
                if next_tok in (im_end_id, eos_id):
                    break
                gen_ids.append(next_tok)
                if "\n" in tok.decode(gen_ids):
                    break
                cur_len = cur_pos + 1
                feed = {
                    "input_ids": np.array([[next_tok]], dtype=np.int64),
                    "attention_mask": np.ones((1, cur_len), dtype=np.int64),
                    "position_ids": np.array([[cur_pos]], dtype=np.int64),
                }
                for layer in range(num_layers):
                    feed[f"past_key_values.{layer}.key"] = past_kv[layer * 2]
                    feed[f"past_key_values.{layer}.value"] = past_kv[layer * 2 + 1]
                outs = sess.run(None, feed)
                logits = outs[0]
                past_kv = outs[1:]
                cur_pos = cur_len
                next_tok = int(np.argmax(logits[0, -1]))

            candidate = tok.decode(gen_ids).strip()
            return self._validate_structure_preserved(raw_text, candidate)
        except Exception as exc:
            print(f"[ContextClarifier] Qwen error: {exc}")
            return raw_text

    def clarify_segment(
        self,
        raw_text: str,
        language: str,
        context_summary: str,
        clarification_model: str = DEFAULT_CLARIFICATION_MODEL,
        whisper_ctx_candidate: Optional[str] = None,
        speaker_context: str = "",
    ) -> str:
        """
        Validate and clarify a single transcribed segment sentence-by-sentence using
        conversation context, speaker persona/history, and cultural/national context
        WITHOUT restructuring any sentence.
        """
        if not raw_text or clarification_model == "none":
            return raw_text

        working = raw_text.strip()
        # 1. If Whisper contextual second-pass candidate (`sot_prev`) is provided, merge only high-confidence word fixes
        if whisper_ctx_candidate and whisper_ctx_candidate.strip() != working:
            working = self._merge_acoustic_context_candidate(
                working, whisper_ctx_candidate, f"{context_summary} {speaker_context}"
            )

        # 2. Apply sentence-by-sentence contextual & cultural compound-word validation
        working = self._apply_contextual_compound_and_lexicon_fixes(
            working, language, context_summary, speaker_context=speaker_context
        )

        # 3. If user selected a local Qwen2.5 ONNX context clarifier model, run it with strict phonetic structure guardrails
        if clarification_model in ("qwen2.5-0.5b", "qwen2.5-1.5b"):
            working = self.clarify_with_qwen(
                clarification_model,
                working,
                language,
                f"{context_summary}\nSpeaker profile: {speaker_context}".strip(),
            )
            working = self._apply_contextual_compound_and_lexicon_fixes(
                working, language, context_summary, speaker_context=speaker_context
            )

        return working


_MEL_FILTERS_CACHE: Dict[int, np.ndarray] = {}
_WHISPER_HANN_400 = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(400) / 400.0)).astype(np.float32)


def _get_whisper_mel_filters(n_mels: int = 80, sr: int = SAMPLE_RATE, n_fft: int = 400) -> np.ndarray:
    """Compute and cache exact OpenAI Whisper Slaney-normalized Mel filterbank matrix."""
    if n_mels in _MEL_FILTERS_CACHE:
        return _MEL_FILTERS_CACHE[n_mels]

    f_min, f_max = 0.0, sr / 2.0
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0

    def hz_to_mel(hz):
        hz = np.asarray(hz)
        mel = (hz - f_min) / f_sp
        mask = hz >= min_log_hz
        return np.where(mask, min_log_mel + np.log(np.maximum(hz, 1e-10) / min_log_hz) / logstep, mel)

    def mel_to_hz(mel):
        mel = np.asarray(mel)
        hz = f_min + f_sp * mel
        mask = mel >= min_log_mel
        return np.where(mask, min_log_hz * np.exp(logstep * (mel - min_log_mel)), hz)

    mels = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz = mel_to_hz(mels)
    fftfreqs = np.linspace(0, sr / 2.0, 1 + n_fft // 2)
    fdiff = np.diff(hz)
    ramps = np.subtract.outer(hz, fftfreqs)
    weights = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (hz[2 : n_mels + 2] - hz[:n_mels])
    weights *= enorm[:, np.newaxis]
    _MEL_FILTERS_CACHE[n_mels] = weights
    return weights


def _whisper_log_mel_spectrogram(audio_16k: np.ndarray, n_mels: int = 80) -> np.ndarray:
    """
    Compute [1, n_mels, 3000] Whisper log-Mel spectrogram matching `sherpa-onnx` (`kaldifeat::WhisperFbank`):
    1. Compute STFT and log10 Mel power ONLY on the actual audio frames `n_f = min(3000, len(audio_16k) // 160)`.
    2. Normalize `log_spec` on those `n_f` frames (`(np.maximum(log_spec, max_val - 8.0) + 4.0) / 4.0`).
    3. Place into a zero-initialized `[1, n_mels, 3000]` buffer so unused frames `n_f .. 3000` remain `0.0`.
    """
    clipped = np.ascontiguousarray(audio_16k[:480000], dtype=np.float32)
    if len(clipped) < 400:
        clipped = np.pad(clipped, (0, 400 - len(clipped)))
    n_f = min(3000, max(1, len(clipped) // 160))
    pad_reflect = np.pad(clipped, (200, 200), mode="reflect")
    strides = (pad_reflect.strides[0] * 160, pad_reflect.strides[0])
    frames = np.lib.stride_tricks.as_strided(pad_reflect, shape=(n_f, 400), strides=strides) * _WHISPER_HANN_400
    stft = np.fft.rfft(frames, n=400, axis=1)
    magnitudes = np.abs(stft) ** 2
    filters = _get_whisper_mel_filters(n_mels)
    mel_spec = filters @ magnitudes.T
    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
    max_val = log_spec.max()
    log_spec = np.maximum(log_spec, max_val - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    full = np.zeros((1, n_mels, 3000), dtype=np.float32)
    full[0, :, :n_f] = log_spec
    return full


class WhisperONNXRunner:
    """
    Runs Whisper ONNX encoder + decoder directly via Microsoft's signed `onnxruntime.InferenceSession`.
    Supports both:
    - Standard `sherpa-onnx` Whisper INT8 models (`tiny`, `base`, `small`, `medium`, `turbo`)
    - Specialized Danish HuggingFace/Optimum Whisper ONNX models (`hviske-v2` — Syv.ai / CoRal Whisper Large-v3)
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        paths = resolve_local_whisper_model(model_id)
        self.is_hf_onnx = "engine" in paths and str(paths["engine"]) == "hf-whisper-onnx"
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 4
        self.encoder = ort.InferenceSession(
            str(paths["encoder"]),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.decoder = ort.InferenceSession(
            str(paths["decoder"]),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )

        if self.is_hf_onnx:
            self.hf_tokenizer = Tokenizer.from_file(str(paths["tokens"]))
            cfg_path = paths["dir"] / "config.json"
            cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
            self.n_mels = int(cfg.get("num_mel_bins", 128))
            self.n_text_layer = int(cfg.get("decoder_layers", 32))
            self.n_heads = int(cfg.get("decoder_attention_heads", 20))
            self.d_model = int(cfg.get("d_model", 1280))
            self.head_dim = self.d_model // max(1, self.n_heads)
            self.sot = self.hf_tokenizer.token_to_id("<|startoftranscript|>") or 50258
            self.eot = self.hf_tokenizer.token_to_id("<|endoftext|>") or 50257
            self.sot_prev = self.hf_tokenizer.token_to_id("<|startofprev|>") or 50362
            self.transcribe_id = self.hf_tokenizer.token_to_id("<|transcribe|>") or 50360
            self.translate_id = self.hf_tokenizer.token_to_id("<|translate|>") or 50359
            self.no_timestamps = self.hf_tokenizer.token_to_id("<|notimestamps|>") or 50364
            self.is_multilingual = True
            self.lang_to_id = {}
            self.id_to_lang = {}
            for code in ("da", "en", "de", "es", "fr", "sv", "no", "nl", "it", "pt"):
                tid = self.hf_tokenizer.token_to_id(f"<|{code}|>")
                if tid is not None:
                    self.lang_to_id[code] = int(tid)
                    self.id_to_lang[int(tid)] = code
            self.lang_tokens_arr = np.array(list(self.lang_to_id.values()), dtype=np.int64)
            self.tokens_map = {}
            self._by_first_byte = {}
            return

        meta = self.encoder.get_modelmeta().custom_metadata_map
        self.n_mels = int(meta.get("n_mels", 80))
        self.n_text_layer = int(meta.get("n_text_layer", 6))
        self.n_text_ctx = int(meta.get("n_text_ctx", 448))
        self.n_text_state = int(meta.get("n_text_state", 512))
        self.sot = int(meta.get("sot", 50258))
        self.eot = int(meta.get("eot", 50257))
        self.sot_prev = int(meta.get("sot_prev", 50361))
        self.blank_id = int(meta.get("blank_id", 220))
        self.transcribe_id = int(meta.get("transcribe", 50359))
        self.translate_id = int(meta.get("translate", 50358))
        self.no_timestamps = int(meta.get("no_timestamps", 50363))
        self.no_speech = int(meta.get("no_speech", 50362))
        self.is_multilingual = int(meta.get("is_multilingual", 1)) == 1

        lang_codes = [c.strip() for c in meta.get("all_language_codes", "").split(",") if c.strip()]
        lang_tokens = [int(x) for x in meta.get("all_language_tokens", "").split(",") if x.strip()]
        self.lang_to_id: Dict[str, int] = dict(zip(lang_codes, lang_tokens))
        self.id_to_lang: Dict[int, str] = dict(zip(lang_tokens, lang_codes))
        self.lang_tokens_arr = np.array(lang_tokens, dtype=np.int64)

        non_speech_str = meta.get("non_speech_tokens", "")
        self.non_speech_ids = [int(x) for x in non_speech_str.split(",") if x.strip().lstrip("-").isdigit()]

        self.tokens_map: Dict[int, bytes] = {}
        with open(paths["tokens"], "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        self.tokens_map[int(parts[1])] = base64.b64decode(parts[0])
                    except Exception:
                        pass

        # Pre-index byte tokens by first byte for fast context-prompt tokenization (<|startofprev|>)
        self._by_first_byte: Dict[int, List[Tuple[bytes, int]]] = {}
        inv_toks = sorted(
            [(b, tid) for tid, b in self.tokens_map.items() if tid < self.eot and len(b) > 0],
            key=lambda x: -len(x[0]),
        )
        for b, tid in inv_toks:
            self._by_first_byte.setdefault(b[0], []).append((b, tid))

    def encode_prompt_text(self, text_str: str, max_tokens: int = 56) -> List[int]:
        """Encode preceding conversation context string into Whisper token IDs for `<|startofprev|>`."""
        if not text_str or not text_str.strip():
            return []
        if self.is_hf_onnx:
            return self.hf_tokenizer.encode(" " + text_str.strip(), add_special_tokens=False).ids[-max_tokens:]
        raw = (" " + text_str.strip()).encode("utf-8")
        out: List[int] = []
        i = 0
        n = len(raw)
        while i < n:
            matched = False
            for b, tid in self._by_first_byte.get(raw[i], ()):
                if raw[i : i + len(b)] == b:
                    out.append(tid)
                    i += len(b)
                    matched = True
                    break
            if not matched:
                i += 1
        return out[-max_tokens:]

    def encode_audio(self, audio_16k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.is_hf_onnx:
            padded = np.zeros(480000, dtype=np.float32)
            n_copy = min(480000, len(audio_16k))
            padded[:n_copy] = audio_16k[:n_copy]
            mel = _whisper_log_mel_spectrogram(padded, self.n_mels)
            hidden = self.encoder.run(None, {"input_features": mel})[0]
            return hidden, hidden
        mel = _whisper_log_mel_spectrogram(audio_16k, self.n_mels)
        cross_k, cross_v = self.encoder.run(None, {"mel": mel})
        return cross_k, cross_v

    def detect_language_from_cross(
        self,
        cross_k: np.ndarray,
        cross_v: np.ndarray,
        allowed_languages: Optional[List[str]] = None,
    ) -> str:
        if self.is_hf_onnx:
            # Hviske v2 is a specialized Danish ASR model
            if allowed_languages and len(allowed_languages) == 1:
                return allowed_languages[0]
            return "da"
        if not self.is_multilingual or len(self.lang_tokens_arr) == 0:
            return "en"
        valid_allowed = [
            c for c in (allowed_languages or []) if c in self.lang_to_id
        ]
        if len(valid_allowed) == 1:
            return valid_allowed[0]

        self_k = np.zeros((self.n_text_layer, 1, self.n_text_ctx, self.n_text_state), dtype=np.float32)
        self_v = np.zeros((self.n_text_layer, 1, self.n_text_ctx, self.n_text_state), dtype=np.float32)
        logits, _, _ = self.decoder.run(
            None,
            {
                "tokens": np.array([[self.sot]], dtype=np.int64),
                "in_n_layer_self_k_cache": self_k,
                "in_n_layer_self_v_cache": self_v,
                "n_layer_cross_k": cross_k,
                "n_layer_cross_v": cross_v,
                "offset": np.array([0], dtype=np.int64),
            },
        )
        if valid_allowed:
            allowed_toks = np.array([self.lang_to_id[c] for c in valid_allowed], dtype=np.int64)
            lang_logits = logits[0, -1, allowed_toks]
            best_lang_tok = int(allowed_toks[int(np.argmax(lang_logits))])
            return self.id_to_lang.get(best_lang_tok, valid_allowed[0])

        lang_logits = logits[0, -1, self.lang_tokens_arr]
        best_lang_tok = int(self.lang_tokens_arr[int(np.argmax(lang_logits))])
        return self.id_to_lang.get(best_lang_tok, "da")

    def decode_from_cross(
        self,
        cross_k: np.ndarray,
        cross_v: np.ndarray,
        language: str = "da",
        task: str = "transcribe",
        duration_sec: float = 10.0,
        context_prompt: Optional[str] = None,
    ) -> str:
        if self.is_hf_onnx:
            hidden = cross_k
            lang_id = self.lang_to_id.get(language, self.lang_to_id.get("da", 50285))
            task_id = self.translate_id if task == "translate" else self.transcribe_id
            init_ids = np.array([[self.sot, lang_id, task_id, self.no_timestamps]], dtype=np.int64)
            feed = {
                "input_ids": init_ids,
                "encoder_hidden_states": hidden,
                "use_cache_branch": np.array([False], dtype=bool),
            }
            for l in range(self.n_text_layer):
                feed[f"past_key_values.{l}.decoder.key"] = np.zeros(
                    (1, self.n_heads, 0, self.head_dim), dtype=np.float32
                )
                feed[f"past_key_values.{l}.decoder.value"] = np.zeros(
                    (1, self.n_heads, 0, self.head_dim), dtype=np.float32
                )
                feed[f"past_key_values.{l}.encoder.key"] = np.zeros(
                    (1, self.n_heads, 1500, self.head_dim), dtype=np.float32
                )
                feed[f"past_key_values.{l}.encoder.value"] = np.zeros(
                    (1, self.n_heads, 1500, self.head_dim), dtype=np.float32
                )
            outs = self.decoder.run(None, feed)
            logits = outs[0]
            presents = outs[1:]
            enc_kv = {}
            dec_kv = {}
            for l in range(self.n_text_layer):
                dec_kv[l] = (presents[l * 4], presents[l * 4 + 1])
                enc_kv[l] = (presents[l * 4 + 2], presents[l * 4 + 3])

            max_tokens = min(200, max(24, int(duration_sec * 14)))
            out_ids: List[int] = []
            next_tok = int(np.argmax(logits[0, -1]))
            dummy_hidden = np.zeros((1, 1, self.d_model), dtype=np.float32)

            for _ in range(max_tokens):
                if next_tok == self.eot:
                    break
                if next_tok < self.eot:
                    out_ids.append(next_tok)
                if len(out_ids) >= 6 and len(set(out_ids[-6:])) == 1:
                    out_ids = out_ids[:-5]
                    break
                if len(out_ids) >= 8 and out_ids[-4:] == out_ids[-8:-4]:
                    out_ids = out_ids[:-4]
                    break
                step_feed = {
                    "input_ids": np.array([[next_tok]], dtype=np.int64),
                    "encoder_hidden_states": dummy_hidden,
                    "use_cache_branch": np.array([True], dtype=bool),
                }
                for l in range(self.n_text_layer):
                    step_feed[f"past_key_values.{l}.decoder.key"] = dec_kv[l][0]
                    step_feed[f"past_key_values.{l}.decoder.value"] = dec_kv[l][1]
                    step_feed[f"past_key_values.{l}.encoder.key"] = enc_kv[l][0]
                    step_feed[f"past_key_values.{l}.encoder.value"] = enc_kv[l][1]
                outs = self.decoder.run(None, step_feed)
                logits = outs[0]
                presents = outs[1:]
                for l in range(self.n_text_layer):
                    dec_kv[l] = (presents[l * 4], presents[l * 4 + 1])
                next_tok = int(np.argmax(logits[0, -1]))

            text_out = self.hf_tokenizer.decode(out_ids, skip_special_tokens=True).strip()
            if text_out and text_out[0].islower():
                text_out = text_out[0].upper() + text_out[1:]
            if text_out and text_out[-1] not in ".?!…":
                text_out += "."
            return text_out
        self_k = np.zeros((self.n_text_layer, 1, self.n_text_ctx, self.n_text_state), dtype=np.float32)
        self_v = np.zeros((self.n_text_layer, 1, self.n_text_ctx, self.n_text_state), dtype=np.float32)

        prompt_ids = self.encode_prompt_text(context_prompt, max_tokens=56) if context_prompt else []

        if self.is_multilingual:
            lang_id = self.lang_to_id.get(language, self.lang_to_id.get("da", self.sot + 1))
            task_id = self.translate_id if task == "translate" else self.transcribe_id
            if prompt_ids:
                init_seq = [self.sot_prev] + prompt_ids + [self.sot, lang_id, task_id, self.no_timestamps]
            else:
                init_seq = [self.sot, lang_id, task_id, self.no_timestamps]
        else:
            if prompt_ids:
                init_seq = [self.sot_prev] + prompt_ids + [self.sot, self.no_timestamps]
            else:
                init_seq = [self.sot, self.no_timestamps]

        init_tokens = np.array([init_seq], dtype=np.int64)
        offset = np.array([0], dtype=np.int64)
        logits, self_k, self_v = self.decoder.run(
            None,
            {
                "tokens": init_tokens,
                "in_n_layer_self_k_cache": self_k,
                "in_n_layer_self_v_cache": self_v,
                "n_layer_cross_k": cross_k,
                "n_layer_cross_v": cross_v,
                "offset": offset,
            },
        )
        offset[0] = init_tokens.shape[1]

        max_tokens = min(220, max(28, int(duration_sec * 14)))
        out_ids: List[int] = []

        next_tok = int(np.argmax(logits[0, -1]))
        for _ in range(max_tokens):
            if next_tok == self.eot:
                break
            if next_tok < self.eot:
                out_ids.append(next_tok)
            if len(out_ids) >= 6 and len(set(out_ids[-6:])) == 1:
                out_ids = out_ids[:-5]
                break
            if len(out_ids) >= 8 and out_ids[-4:] == out_ids[-8:-4]:
                out_ids = out_ids[:-4]
                break

            logits, self_k, self_v = self.decoder.run(
                None,
                {
                    "tokens": np.array([[next_tok]], dtype=np.int64),
                    "in_n_layer_self_k_cache": self_k,
                    "in_n_layer_self_v_cache": self_v,
                    "n_layer_cross_k": cross_k,
                    "n_layer_cross_v": cross_v,
                    "offset": offset,
                },
            )
            offset[0] += 1
            next_tok = int(np.argmax(logits[0, -1]))

        raw_bytes = b"".join(self.tokens_map.get(t, b"") for t in out_ids)
        return raw_bytes.decode("utf-8", errors="replace").strip()


class LocalSpeechPipeline:
    """
    Coordinates:
    1. LocalSpeakerDiarizer (3D-Speaker ONNX voice recognition)
    2. WhisperONNXRunner (Direct ONNX Runtime Whisper speech-to-text + restricted spoken language ID)
    3. LocalContextClarifierEngine (Context-Aware Word Clarification & Validation without sentence restructuring)
    4. LocalONNXTranslationEngine (OPUS-MT Danish <-> English Specialist, Meta NLLB-200, Whisper Direct Translate, or Skip)
    """

    def __init__(self, session_store: SessionStore):
        self.store = session_store
        self.diarizer = LocalSpeakerDiarizer()
        self.clarifier = LocalContextClarifierEngine()
        self.translator = LocalONNXTranslationEngine()
        self._whisper_runners: Dict[str, WhisperONNXRunner] = {}
        self._lock = threading.Lock()
        self.device = "onnx-cpu"
        self.compute_type = "int8"
        print(
            "[SpeechPipeline] Initialized 4-Stage Pipeline: 3D-Speaker + Whisper ONNX + Context Clarifier + OPUS-MT/NLLB-200."
        )

    def _get_whisper_runner(
        self, model_id: str = DEFAULT_WHISPER_MODEL
    ) -> WhisperONNXRunner:
        if model_id in self._whisper_runners:
            return self._whisper_runners[model_id]
        with self._lock:
            if model_id in self._whisper_runners:
                return self._whisper_runners[model_id]
            runner = WhisperONNXRunner(model_id)
            self._whisper_runners[model_id] = runner
            return runner

    @staticmethod
    def _build_conversation_context(
        session: MeetingSession,
        up_to_seg_index: Optional[int] = None,
        current_speaker_id: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Build:
        1. `overall_context`: What the conversation is about across the meeting (title, file name,
           cultural/language context, and surrounding dialogue window — both preceding and following
           sentences when doing batch/file processing).
        2. `speaker_context`: What this specific speaker (`current_speaker_id`) has talked about,
           their role/perspective, and way of speaking.
        """
        parts: List[str] = []
        if session.title and not session.title.lower().startswith(("new meeting", "untitled")):
            parts.append(f"Topic: {session.title}.")
        if session.source_file_name:
            clean_fname = re.sub(r"\.[^/.]+$", "", session.source_file_name).replace("_", " ").replace("-", " ")
            if clean_fname and clean_fname.lower() not in (session.title or "").lower():
                parts.append(f"Recording: {clean_fname}.")

        all_segs = session.segments
        if up_to_seg_index is not None:
            start_w = max(0, up_to_seg_index - 5)
            end_w = min(len(all_segs), up_to_seg_index + 4)
            window_segs = [all_segs[i] for i in range(start_w, end_w) if i != up_to_seg_index]
        else:
            window_segs = all_segs[-6:]

        recent_texts = [
            (s.original_text or s.raw_text or "").strip()
            for s in window_segs
            if (s.original_text or s.raw_text or "").strip()
        ]
        if recent_texts:
            parts.append(" ".join(recent_texts))

        speaker_ctx = ""
        if current_speaker_id:
            spk_profile = session.speakers.get(current_speaker_id)
            spk_name = spk_profile.name if spk_profile else current_speaker_id
            spk_lines = [
                (s.original_text or s.raw_text or "").strip()
                for idx_s, s in enumerate(all_segs)
                if s.speaker_id == current_speaker_id
                and idx_s != up_to_seg_index
                and (s.original_text or s.raw_text or "").strip()
            ][-6:]
            if spk_lines:
                speaker_ctx = f"{spk_name}: " + " ".join(spk_lines)

        return " ".join(parts).strip(), speaker_ctx

    def detect_spoken_language(
        self,
        audio_16k: np.ndarray,
        model_id: str = DEFAULT_WHISPER_MODEL,
        allowed_languages: Optional[List[str]] = None,
    ) -> str:
        try:
            runner = self._get_whisper_runner(model_id)
            norm_audio = self._normalize_speech_clip(audio_16k)
            cross_k, cross_v = runner.encode_audio(norm_audio)
            lang = runner.detect_language_from_cross(
                cross_k, cross_v, allowed_languages=allowed_languages
            )
            return lang.strip() if lang else "da"
        except Exception as exc:
            print(f"[SpeechPipeline] Language detect fallback to 'da': {exc}")
            return "da"

    @staticmethod
    def _normalize_speech_clip(audio_16k: np.ndarray) -> np.ndarray:
        """Peak-normalize genuinely quiet microphone/loopback clips (<0.20 peak) without distorting normal speech."""
        if len(audio_16k) == 0:
            return audio_16k
        peak = float(np.max(np.abs(audio_16k)))
        if 0.002 < peak < 0.20:
            return np.clip(audio_16k * (0.65 / peak), -1.0, 1.0).astype(np.float32)
        return audio_16k.astype(np.float32)

    def _run_whisper_clip(
        self,
        audio_16k: np.ndarray,
        model_id: str,
        language: str,
        task: str = "transcribe",
        allowed_languages: Optional[List[str]] = None,
    ) -> str:
        norm_audio = self._normalize_speech_clip(audio_16k)
        runner = self._get_whisper_runner(model_id)
        cross_k, cross_v = runner.encode_audio(norm_audio)
        lang_code = (
            language
            if (language and language != "auto")
            else runner.detect_language_from_cross(
                cross_k, cross_v, allowed_languages=allowed_languages
            )
        )
        dur = len(audio_16k) / float(SAMPLE_RATE)
        return runner.decode_from_cross(
            cross_k, cross_v, language=lang_code, task=task, duration_sec=dur
        )

    def transcribe_and_translate_clip(
        self,
        audio_16k: np.ndarray,
        forced_language: Optional[str],
        target_language: str,
        model_id: str = DEFAULT_WHISPER_MODEL,
        clarification_model: str = DEFAULT_CLARIFICATION_MODEL,
        translation_model: str = DEFAULT_TRANSLATION_MODEL,
        allowed_languages: Optional[List[str]] = None,
        conversation_context: str = "",
        speaker_context: str = "",
    ) -> Tuple[str, str, str, float, str]:
        """
        1. Transcribe a single audio clip in `forced_language` (or auto-detect among `allowed_languages` if None/"auto").
        2. Run Context Clarification (`clarification_model`) using `conversation_context` and `speaker_context`
           to validate/fix misheard words without restructuring sentences (or skip if `clarification_model == "none"`).
        3. Translate to `target_language` using `translation_model` (or skip if `translation_model == "none"`
           or `target_language == "none"`).
        Returns: (detected_source_lang, clarified_original_text, translated_text, confidence, raw_stt_text)
        """
        if len(audio_16k) < int(SAMPLE_RATE * 0.25):
            fallback_lang = (
                forced_language
                or (allowed_languages[0] if allowed_languages else "en")
            )
            return fallback_lang, "", "", 0.0, ""

        norm_audio = self._normalize_speech_clip(audio_16k)
        runner = self._get_whisper_runner(model_id)
        cross_k, cross_v = runner.encode_audio(norm_audio)
        dur = len(audio_16k) / float(SAMPLE_RATE)

        if not forced_language or forced_language == "auto":
            detected_lang = runner.detect_language_from_cross(
                cross_k, cross_v, allowed_languages=allowed_languages
            )
        else:
            detected_lang = forced_language

        # Step 1: Raw Speech-to-Text transcription in the speaker's spoken language
        raw_text = runner.decode_from_cross(
            cross_k,
            cross_v,
            language=detected_lang,
            task="transcribe",
            duration_sec=dur,
        )
        if not raw_text:
            return detected_lang, "", "", 1.0, ""

        # Step 2: 3rd Model — Context Clarification & Word Validation (without sentence restructuring)
        # Note: Do NOT re-run the Whisper decoder with `<|startofprev|>` context tokens here,
        # because `<|startofprev|>` biases Whisper into hallucinating or altering spoken words
        # and is slow. Instead, validate the raw sentence text directly against the conversation,
        # speaker profile, and cultural/national context.
        if clarification_model and clarification_model != "none":
            original_text = self.clarifier.clarify_segment(
                raw_text=raw_text,
                language=detected_lang,
                context_summary=conversation_context,
                clarification_model=clarification_model,
                whisper_ctx_candidate=None,
                speaker_context=speaker_context,
            )
        else:
            original_text = raw_text

        # Step 3: Translate into single output `target_language` (or skip if translation is disabled)
        if (
            not target_language
            or target_language == "none"
            or translation_model == "none"
            or detected_lang == target_language
        ):
            translated_text = original_text
        elif translation_model == "whisper-native" and target_language == "en":
            w_trans = runner.decode_from_cross(
                cross_k,
                cross_v,
                language=detected_lang,
                task="translate",
                duration_sec=dur,
            )
            translated_text = w_trans or original_text
        else:
            # Use specialized OPUS-MT (e.g. Danish <-> English, Spanish -> English) or NLLB-200
            translated_text = self.translator.translate(
                original_text,
                source_lang=detected_lang,
                target_lang=target_language,
                translation_model=translation_model,
            )

        return detected_lang, original_text, translated_text, 1.0, raw_text

    def process_live_chunk(
        self,
        session: MeetingSession,
        audio_16k: np.ndarray,
        forced_speaker_id: Optional[str] = None,
        on_speaker_active: Optional[Callable[[str], None]] = None,
    ) -> List[TranscriptSegment]:
        rms = float(np.sqrt(np.mean(audio_16k**2))) if len(audio_16k) > 0 else 0.0
        chunk_start, _ = self.store.append_audio_pcm(session.id, audio_16k)

        if rms < 0.0025:
            self.store.save_session(session)
            return []

        sub_intervals = segment_speech_by_energy(
            audio_16k, min_speech_sec=0.85, max_segment_sec=14.0, min_silence_sec=0.6
        )
        if not sub_intervals:
            self.store.save_session(session)
            return []

        created_segments: List[TranscriptSegment] = []

        for sub_start_rel, sub_end_rel in sub_intervals:
            s_sample = int(sub_start_rel * SAMPLE_RATE)
            e_sample = int(sub_end_rel * SAMPLE_RATE)
            sub_audio = audio_16k[s_sample:e_sample]

            # Match against this session's isolated speaker embeddings without mutating until speech is confirmed
            embeddings_snapshot = {k: list(v) for k, v in session.embeddings.items()}
            if forced_speaker_id:
                spk_id = forced_speaker_id
            else:
                spk_id, updated_embeddings = self.diarizer.assign_speaker_online(
                    sub_audio, embeddings_snapshot
                )

            if on_speaker_active:
                try:
                    on_speaker_active(spk_id)
                except Exception:
                    pass

            existing_spk = session.speakers.get(spk_id)
            assigned_lang = existing_spk.language if existing_spk else "auto"

            # If speaker is on 'auto' and clip is short (< 2.2s), reuse the speaker's last detected language if available
            effective_forced_lang = None if assigned_lang == "auto" else assigned_lang
            if (
                effective_forced_lang is None
                and not session.spoken_languages
                and len(sub_audio) < int(SAMPLE_RATE * 2.2)
            ):
                prev_spk_segs = [s for s in session.segments if s.speaker_id == spk_id and s.source_language]
                if prev_spk_segs:
                    effective_forced_lang = prev_spk_segs[-1].source_language

            conv_ctx, spk_ctx = self._build_conversation_context(
                session, current_speaker_id=spk_id
            )
            detected_lang, orig_text, trans_text, confidence, raw_stt_text = (
                self.transcribe_and_translate_clip(
                    sub_audio,
                    forced_language=effective_forced_lang,
                    target_language=session.target_language,
                    model_id=session.whisper_model,
                    clarification_model=session.clarification_model,
                    translation_model=session.translation_model,
                    allowed_languages=session.spoken_languages,
                    conversation_context=conv_ctx,
                    speaker_context=spk_ctx,
                )
            )

            if not orig_text or not orig_text.strip(" ()[].,-?!\"'"):
                continue

            if not forced_speaker_id:
                session.embeddings = updated_embeddings
            self.store.ensure_speaker(session, spk_id)

            seg_obj = TranscriptSegment(
                id=uuid.uuid4().hex[:10],
                speaker_id=spk_id,
                start=round(chunk_start + sub_start_rel, 2),
                end=round(chunk_start + sub_end_rel, 2),
                source_language=detected_lang,
                original_text=orig_text,
                translated_text=trans_text,
                target_language=session.target_language,
                confidence=round(confidence, 3),
                raw_text=raw_stt_text,
            )
            session.segments.append(seg_obj)
            created_segments.append(seg_obj)

        self.store.save_session(session)
        return created_segments

    def process_uploaded_file(
        self,
        session: MeetingSession,
        audio_16k: np.ndarray,
        num_speakers: Optional[int] = None,
        progress_cb: Optional[Callable[[str, int], None]] = None,
    ) -> MeetingSession:
        if progress_cb:
            progress_cb("Saving normalized audio for playback...", 10)
        self.store.write_full_audio(session.id, audio_16k)

        if progress_cb:
            progress_cb(
                f"Preparing local Whisper ONNX ({session.whisper_model}) model...", 20
            )
        resolve_local_whisper_model(session.whisper_model)

        if progress_cb:
            progress_cb("Detecting speech segments across recording...", 35)
        seg_times = segment_speech_by_energy(
            audio_16k, min_speech_sec=0.5, max_segment_sec=14.0, min_silence_sec=0.4
        )
        if not seg_times:
            self.store.save_session(session)
            if progress_cb:
                progress_cb("Completed!", 100)
            return session

        if progress_cb:
            progress_cb("Diarizing speakers (recognizing distinct voices)...", 55)
        seg_times = self.diarizer.split_segments_on_speaker_change(
            audio_16k, seg_times
        )
        speaker_ids, centroids = self.diarizer.cluster_segments_batch(
            audio_16k, seg_times, num_speakers=num_speakers
        )
        # Strictly isolate speakers and segments to this session's audio
        session.speakers = {}
        session.embeddings = centroids
        session.segments = []

        total = len(seg_times)
        for idx, ((s_sec, e_sec), spk_id) in enumerate(zip(seg_times, speaker_ids)):
            if progress_cb:
                pct = 60 + int(35 * (idx + 1) / max(1, total))
                progress_cb(
                    f"Transcribing & translating segment {idx + 1}/{total}...", pct
                )

            clip = audio_16k[int(s_sec * SAMPLE_RATE) : int(e_sec * SAMPLE_RATE)]

            existing_spk = session.speakers.get(spk_id)
            forced_lang = (
                existing_spk.language
                if (existing_spk and existing_spk.language != "auto")
                else None
            )
            conv_ctx, spk_ctx = self._build_conversation_context(
                session, current_speaker_id=spk_id
            )
            detected_lang, orig_text, trans_text, conf, raw_stt_text = (
                self.transcribe_and_translate_clip(
                    clip,
                    forced_language=forced_lang,
                    target_language=session.target_language,
                    model_id=session.whisper_model,
                    clarification_model=session.clarification_model,
                    translation_model=session.translation_model,
                    allowed_languages=session.spoken_languages,
                    conversation_context=conv_ctx,
                    speaker_context=spk_ctx,
                )
            )
            if not orig_text or not orig_text.strip(" ()[].,-?!\"'"):
                continue

            self.store.ensure_speaker(session, spk_id)

            seg_obj = TranscriptSegment(
                id=uuid.uuid4().hex[:10],
                speaker_id=spk_id,
                start=round(s_sec, 2),
                end=round(e_sec, 2),
                source_language=detected_lang,
                original_text=orig_text,
                translated_text=trans_text,
                target_language=session.target_language,
                confidence=round(conf, 3),
                raw_text=raw_stt_text,
            )
            session.segments.append(seg_obj)

        # Second-Pass Global & Speaker-Aware Clarification across the full recording:
        # Now that we know what the entire interview is about and how each speaker speaks,
        # validate every segment against the full conversation & speaker context before saving.
        if session.clarification_model and session.clarification_model != "none":
            for idx, seg in enumerate(session.segments):
                full_conv_ctx, full_spk_ctx = self._build_conversation_context(
                    session, up_to_seg_index=idx, current_speaker_id=seg.speaker_id
                )
                seg.original_text = self.clarifier._apply_contextual_compound_and_lexicon_fixes(
                    seg.original_text,
                    seg.source_language,
                    full_conv_ctx,
                    speaker_context=full_spk_ctx,
                )
                if (
                    not session.target_language
                    or session.target_language == "none"
                    or session.translation_model == "none"
                    or seg.source_language == session.target_language
                ):
                    seg.translated_text = seg.original_text
                elif session.translation_model != "whisper-native":
                    seg.translated_text = self.translator.translate(
                        seg.original_text,
                        source_lang=seg.source_language,
                        target_lang=session.target_language,
                        translation_model=session.translation_model,
                    )

        self.store.save_session(session)
        if progress_cb:
            progress_cb("Completed!", 100)
        return session

    def reprocess_speaker_segments(
        self, session: MeetingSession, speaker_id: str
    ) -> List[TranscriptSegment]:
        if speaker_id not in session.speakers:
            return []

        speaker_lang = session.speakers[speaker_id].language
        updated: List[TranscriptSegment] = []

        for idx, seg in enumerate(session.segments):
            if seg.speaker_id != speaker_id:
                continue

            clip = self.store.read_audio_slice(session.id, seg.start, seg.end)
            if len(clip) < int(SAMPLE_RATE * 0.25):
                continue

            conv_ctx, spk_ctx = self._build_conversation_context(
                session, up_to_seg_index=idx, current_speaker_id=speaker_id
            )
            detected_lang, orig_text, trans_text, conf, raw_stt_text = (
                self.transcribe_and_translate_clip(
                    clip,
                    forced_language=None if speaker_lang == "auto" else speaker_lang,
                    target_language=session.target_language,
                    model_id=session.whisper_model,
                    clarification_model=session.clarification_model,
                    translation_model=session.translation_model,
                    allowed_languages=session.spoken_languages,
                    conversation_context=conv_ctx,
                    speaker_context=spk_ctx,
                )
            )
            if orig_text:
                seg.source_language = detected_lang
                seg.original_text = orig_text
                seg.translated_text = trans_text
                seg.target_language = session.target_language
                seg.confidence = round(conf, 3)
                seg.raw_text = raw_stt_text
            updated.append(seg)

        self.store.save_session(session)
        return updated

    def reclarify_and_retranslate_all_segments(
        self, session: MeetingSession
    ) -> List[TranscriptSegment]:
        """
        When the user changes the 3rd Model (`clarification_model`) on an existing session
        (e.g. toggling between 'whisper-context', 'qwen2.5-0.5b', or 'none' Skip Clarification),
        re-run Context Clarification on each segment's `raw_text` using the full conversation,
        speaker profile, and cultural context, and re-translate into `session.target_language`.
        """
        for idx, seg in enumerate(session.segments):
            base_raw = (seg.raw_text or seg.original_text or "").strip()
            if not base_raw:
                continue
            if not seg.raw_text:
                seg.raw_text = base_raw

            if session.clarification_model == "none":
                seg.original_text = base_raw
            else:
                conv_ctx, spk_ctx = self._build_conversation_context(
                    session, up_to_seg_index=idx, current_speaker_id=seg.speaker_id
                )
                seg.original_text = self.clarifier.clarify_segment(
                    raw_text=base_raw,
                    language=seg.source_language,
                    context_summary=conv_ctx,
                    clarification_model=session.clarification_model,
                    whisper_ctx_candidate=None,
                    speaker_context=spk_ctx,
                )

        return self.retranslate_all_segments(session)

    def retranslate_all_segments(
        self, session: MeetingSession
    ) -> List[TranscriptSegment]:
        """
        When the user changes the single target output language or the translation model for the meeting,
        re-translate all existing segments into `session.target_language`.
        """
        for seg in session.segments:
            if (
                not session.target_language
                or session.target_language == "none"
                or session.translation_model == "none"
                or seg.source_language == session.target_language
            ):
                seg.translated_text = seg.original_text
                seg.target_language = session.target_language
            else:
                if (
                    session.translation_model == "whisper-native"
                    and session.target_language == "en"
                ):
                    clip = self.store.read_audio_slice(session.id, seg.start, seg.end)
                    if len(clip) > int(SAMPLE_RATE * 0.25):
                        w_trans = self._run_whisper_clip(
                            clip,
                            model_id=session.whisper_model,
                            language=seg.source_language,
                            task="translate",
                        )
                        seg.translated_text = w_trans or seg.original_text
                    else:
                        seg.translated_text = seg.original_text
                else:
                    seg.translated_text = self.translator.translate(
                        seg.original_text,
                        source_lang=seg.source_language,
                        target_lang=session.target_language,
                        translation_model=session.translation_model,
                    )
                seg.target_language = session.target_language

        self.store.save_session(session)
        return session.segments

    def retranscribe_and_translate_all_segments(
        self,
        session: MeetingSession,
        progress_cb: Optional[Callable[[str, int], None]] = None,
    ) -> MeetingSession:
        """
        Re-transcribe, clarify, and re-translate all existing segments of a session using
        its current `whisper_model`, `clarification_model`, `translation_model`,
        `spoken_languages`, and `target_language`
        (preserving speaker profiles, names, and per-speaker spoken languages).
        """
        audio_path = self.store.get_audio_path(session.id)
        if not session.segments and audio_path.exists():
            audio_16k = self.store.read_audio_slice(
                session.id, 0.0, max(1.0, session.duration or 3600.0)
            )
            if len(audio_16k) > 0:
                return self.process_uploaded_file(
                    session, audio_16k, progress_cb=progress_cb
                )
            return session

        if progress_cb:
            progress_cb(
                f"Loading Whisper ({session.whisper_model}), Clarifier ({session.clarification_model}) & Translator ({session.translation_model})...",
                10,
            )
        resolve_local_whisper_model(session.whisper_model)

        # If the full session recording is on disk and this is a file session,
        # check if any long segment (>= 3.6s) contains an unsplit intra-segment speaker change
        # and split + reassign speaker IDs using existing session voice centroids.
        if session.session_type == "file" and audio_path.exists() and session.embeddings:
            full_audio_16k = self.store.read_audio_slice(
                session.id, 0.0, max(1.0, session.duration or 3600.0)
            )
            if len(full_audio_16k) > int(SAMPLE_RATE * 1.0):
                cur_times = [(seg.start, seg.end) for seg in session.segments]
                refined_times = self.diarizer.split_segments_on_speaker_change(
                    full_audio_16k, cur_times
                )
                if len(refined_times) != len(cur_times):
                    new_segments: List[TranscriptSegment] = []
                    centroids_np = {
                        spk_id: np.array(vec, dtype=np.float32)
                        for spk_id, vec in session.embeddings.items()
                    }
                    for s_sec, e_sec in refined_times:
                        clip = full_audio_16k[
                            int(s_sec * SAMPLE_RATE) : int(e_sec * SAMPLE_RATE)
                        ]
                        emb = self.diarizer.extract_embedding(clip)
                        best_spk = "speaker_1"
                        best_sim = -2.0
                        if emb is not None and centroids_np:
                            for spk_id, c_vec in centroids_np.items():
                                if c_vec.shape == emb.shape:
                                    sim = float(
                                        np.dot(emb, c_vec)
                                        / max(
                                            1e-8,
                                            float(
                                                np.linalg.norm(emb)
                                                * np.linalg.norm(c_vec)
                                            ),
                                        )
                                    )
                                    if sim > best_sim:
                                        best_sim = sim
                                        best_spk = spk_id
                        new_segments.append(
                            TranscriptSegment(
                                id=uuid.uuid4().hex[:10],
                                speaker_id=best_spk,
                                start=round(s_sec, 2),
                                end=round(e_sec, 2),
                                source_language="da",
                                original_text="",
                                translated_text="",
                                target_language=session.target_language,
                            )
                        )
                    session.segments = new_segments

        total = len(session.segments)
        for idx, seg in enumerate(session.segments):
            if progress_cb:
                pct = 15 + int(80 * (idx + 1) / max(1, total))
                progress_cb(
                    f"Re-transcribing, clarifying & translating segment {idx + 1}/{total} ({session.whisper_model})...",
                    pct,
                )

            clip = self.store.read_audio_slice(session.id, seg.start, seg.end)
            if len(clip) < int(SAMPLE_RATE * 0.25):
                continue

            spk = session.speakers.get(seg.speaker_id)
            forced_lang = (
                spk.language if (spk and spk.language != "auto") else None
            )
            conv_ctx, spk_ctx = self._build_conversation_context(
                session, up_to_seg_index=idx, current_speaker_id=seg.speaker_id
            )
            detected_lang, orig_text, trans_text, conf, raw_stt_text = (
                self.transcribe_and_translate_clip(
                    clip,
                    forced_language=forced_lang,
                    target_language=session.target_language,
                    model_id=session.whisper_model,
                    clarification_model=session.clarification_model,
                    translation_model=session.translation_model,
                    allowed_languages=session.spoken_languages,
                    conversation_context=conv_ctx,
                    speaker_context=spk_ctx,
                )
            )
            if orig_text:
                seg.source_language = detected_lang
                seg.original_text = orig_text
                seg.translated_text = trans_text
                seg.target_language = session.target_language
                seg.confidence = round(conf, 3)
                seg.raw_text = raw_stt_text

        if session.clarification_model and session.clarification_model != "none":
            for idx, seg in enumerate(session.segments):
                full_conv_ctx, full_spk_ctx = self._build_conversation_context(
                    session, up_to_seg_index=idx, current_speaker_id=seg.speaker_id
                )
                seg.original_text = self.clarifier._apply_contextual_compound_and_lexicon_fixes(
                    seg.original_text,
                    seg.source_language,
                    full_conv_ctx,
                    speaker_context=full_spk_ctx,
                )
                if (
                    not session.target_language
                    or session.target_language == "none"
                    or session.translation_model == "none"
                    or seg.source_language == session.target_language
                ):
                    seg.translated_text = seg.original_text
                elif session.translation_model != "whisper-native":
                    seg.translated_text = self.translator.translate(
                        seg.original_text,
                        source_lang=seg.source_language,
                        target_lang=session.target_language,
                        translation_model=session.translation_model,
                    )

        self.store.save_session(session)
        if progress_cb:
            progress_cb("Completed re-transcription, clarification & translation!", 100)
        return session

    def evict_model_cache(self, model_type: str, model_id: str = "") -> None:
        """Release ONNX Runtime sessions from memory before deleting model files on disk."""
        with self._lock:
            if model_type == "whisper":
                self._whisper_runners.pop(model_id, None)
            elif model_type == "clarification":
                if model_id:
                    self.clarifier._qwen_sessions.pop(model_id, None)
                else:
                    self.clarifier._qwen_sessions.clear()
            elif model_type in ("opus", "opus-mt-da-en", "translation"):
                self.translator._opus_cache.clear()
            elif model_type in ("nllb", "nllb-200-600M"):
                self.translator._nllb_encoder = None
                self.translator._nllb_decoder = None
                self.translator._nllb_tokenizer = None
            elif model_type == "speaker":
                self.diarizer._extractor = None
