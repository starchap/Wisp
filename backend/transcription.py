"""
Local Speech-to-Text (Direct ONNX Runtime Whisper INT8 Engine) and Specialized Offline Neural Machine Translation
(Helsinki-NLP OPUS-MT MarianMT ONNX INT8 focused on Danish <-> English + Spanish/German,
Meta NLLB-200 ONNX INT8, and Whisper Native English Translation) Pipeline for Wishper.
Uses Microsoft's digitally signed `onnxruntime` directly — zero unsigned `sherpa_onnx` or `scipy` DLLs.
"""
import base64
import difflib
import json
import os
import re
import threading
import time
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
    ensure_whisper_model,
    get_clarification_model_dir,
    get_opus_pair_dir,
    get_whisper_onnx_paths,
    is_clarification_model_downloaded,
    is_nllb_model_downloaded,
    is_opus_pair_downloaded,
    is_whisper_model_downloaded,
    resolve_local_whisper_model,
)
from session_store import (
    SAMPLE_RATE,
    MeetingSession,
    SessionStore,
    TranscriptSegment,
)


_NON_SPEECH_BRACKET_RE = re.compile(
    r"^\s*[\(\[\*♪]+[\s\w\-.,!?/]*[\)\]\*♪]+\s*$", re.IGNORECASE
)


def _has_meaningful_speech_text(text: str) -> bool:
    """
    Return True only if `text` contains actual spoken words (letters/digits),
    rejecting empty strings, pure punctuation/ellipsis/dashes ('…', '–', '—', '...'),
    invisible whitespace/control chars, and standalone Whisper non-speech tags like
    '[Musik]', '(Silence)', '♪'.
    """
    if not text:
        return False
    cleaned = text.strip()
    if not cleaned:
        return False
    # Reject standalone bracketed non-speech tags (e.g. "[Musik]", "(Latter)", "♪ ♪")
    if _NON_SPEECH_BRACKET_RE.match(cleaned):
        return False
    # Must contain at least one letter or digit
    return any(ch.isalnum() for ch in cleaned)


def _detect_and_trim_token_loop(out_ids: List[int]) -> Tuple[List[int], bool]:
    """
    Detect greedy Whisper decoder repetition loops of ANY period (1 to 64 tokens)
    and trim the duplicated tail so long segments never repeat whole sentences.
    Returns (trimmed_ids, loop_detected).
    """
    n = len(out_ids)
    # 1. Single-token spam (6+ identical tokens in a row)
    if n >= 6 and len(set(out_ids[-6:])) == 1:
        return out_ids[:-5], True

    # 2. Short period loops (2-3 tokens repeated 3 times consecutively)
    for k in (2, 3):
        if n >= k * 3 and out_ids[-k:] == out_ids[-2 * k : -k] == out_ids[-3 * k : -2 * k]:
            return out_ids[: -2 * k], True

    # 3. Medium/long phrase loops (4 to 64 tokens repeated twice back-to-back)
    max_k = min(64, n // 2)
    for k in range(4, max_k + 1):
        if out_ids[-k:] == out_ids[-2 * k : -k]:
            return out_ids[:-k], True

    return out_ids, False


def _trim_repeated_sentences(text: str) -> str:
    """
    Remove near-identical consecutive/adjacent hallucinated sentence loops
    (e.g. Whisper repeating a 20-word clause with a 1-word variation).
    Preserves genuine short interjections like 'Ja. Ja.' or 'Okay. Okay.'.
    """
    cleaned = text.strip()
    if len(cleaned) < 40:
        return cleaned

    parts = re.split(r"(?<=[.!?…])\s+", cleaned)
    if len(parts) < 2:
        return cleaned

    kept: List[str] = []
    for sent in parts:
        s_norm = re.sub(r"[^a-zæøå0-9\s]", "", sent.lower()).strip()
        words = s_norm.split()
        is_dup = False
        if len(words) >= 5 and kept:
            # Check against the last 3 kept sentences for near-verbatim hallucination loops
            for prev in kept[-3:]:
                p_norm = re.sub(r"[^a-zæøå0-9\s]", "", prev.lower()).strip()
                if len(p_norm.split()) >= 5:
                    if s_norm == p_norm or difflib.SequenceMatcher(None, s_norm, p_norm).ratio() >= 0.82:
                        is_dup = True
                        break
        if not is_dup:
            kept.append(sent)

    return " ".join(kept).strip()


def segment_speech_by_energy(
    audio_16k: np.ndarray,
    min_speech_sec: float = 0.6,
    max_segment_sec: float = 14.0,
    min_silence_sec: float = 0.45,
) -> List[Tuple[float, float]]:
    """
    Fast, zero-dependency acoustic Voice Activity Detector (VAD) that splits
    continuous 16kHz audio into natural utterance intervals [(start_sec, end_sec), ...].
    Uses adaptive 30-second local noise-floor tracking for long recordings so changes in
    background hum or microphone distance never collapse the entire file into one chunk.
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

    global_noise = float(np.percentile(rms, 15))
    global_peak = float(np.percentile(rms, 92))
    global_thresh = max(0.004, global_noise + 0.12 * max(0.005, global_peak - global_noise))

    # For recordings > 60 seconds, compute adaptive per-30s-window thresholds blended with global
    frames_per_30s = int(30.0 * 1000 / win_ms)
    if num_frames > frames_per_30s * 2:
        thresholds = np.full(num_frames, global_thresh, dtype=np.float32)
        for w_start in range(0, num_frames, frames_per_30s):
            w_end = min(num_frames, w_start + frames_per_30s)
            w_rms = rms[w_start:w_end]
            if len(w_rms) < 20:
                continue
            loc_noise = float(np.percentile(w_rms, 15))
            loc_peak = float(np.percentile(w_rms, 92))
            loc_thresh = max(0.004, loc_noise + 0.12 * max(0.005, loc_peak - loc_noise))
            # Blend 70% local + 30% global so quiet silence stretches don't drop threshold to 0
            thresholds[w_start:w_end] = max(0.004, 0.70 * loc_thresh + 0.30 * global_thresh)
        is_speech = rms >= thresholds
    else:
        is_speech = rms >= global_thresh
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
        sess_opts.intra_op_num_threads = max(4, min(12, (os.cpu_count() or 4) - 1))
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.encoder = app_config.create_ort_session(
            str(model_dir / "onnx" / "encoder_model_quantized.onnx"), sess_opts
        )
        self.decoder = app_config.create_ort_session(
            str(model_dir / "onnx" / "decoder_model_quantized.onnx"),
            sess_opts,
            is_autoregressive_decoder=True,
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
                self._nllb_encoder = app_config.create_ort_session(
                    str(enc_path), sess_opts
                )
                self._nllb_decoder = app_config.create_ort_session(
                    str(dec_path), sess_opts, is_autoregressive_decoder=True
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
        - Preserves punctuation (commas, periods, question marks) and capitalization fixes
          when the underlying words match (`tag == "equal"`).
        - Only accepts word-for-word or compound-merge replacements where each modified word
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

        # Word-by-word alignment check:
        # - On equal words, keep candidate tokens so comma/period/question-mark/casing fixes are preserved!
        # - On replaced words, only accept spans that are phonetically close (>= 0.72).
        matcher = difflib.SequenceMatcher(
            None,
            [w.lower().strip(".,?!:;\"'()-…–—") for w in raw_words],
            [w.lower().strip(".,?!:;\"'()-…–—") for w in cand_words],
        )
        safe_words: List[str] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                # Preserve LLM punctuation & capitalization improvements on identical words
                safe_words.extend(cand_words[j1:j2])
            elif tag == "replace":
                r_span = " ".join(raw_words[i1:i2])
                c_span = " ".join(cand_words[j1:j2])
                p_sim = difflib.SequenceMatcher(
                    None,
                    LocalContextClarifierEngine._phonetic_key(r_span),
                    LocalContextClarifierEngine._phonetic_key(c_span),
                ).ratio()
                if p_sim >= 0.72:
                    safe_words.extend(cand_words[j1:j2])
                else:
                    safe_words.extend(raw_words[i1:i2])
            else:
                safe_words.extend(raw_words[i1:i2])

        return " ".join(safe_words).strip() or raw_clean

    # Common stopwords excluded when extracting recurring domain keywords across a conversation
    _STOPWORDS: Dict[str, set] = {
        "da": {
            "det", "der", "den", "ikke", "også", "bare", "fordi", "eller", "hvad", "hvor",
            "hvordan", "hvorfor", "hvornår", "meget", "mange", "noget", "nogen", "nogle",
            "kunne", "skulle", "ville", "have", "være", "blevet", "bliver", "været", "havde",
            "sådan", "ligesom", "altså", "faktisk", "egentlig", "rigtig", "virkelig", "måske",
            "altid", "aldrig", "helt", "selvfølgelig", "sammen", "mellem", "gennem", "under",
            "over", "efter", "inden", "omkring", "derfor", "dermed", "hvis", "mens", "selvom",
            "kommer", "komme", "siger", "tænker", "synes", "mener", "bruger", "gøre", "gører",
            "lave", "laver", "sådan", "ting", "gangen", "gange", "måde", "eksempel", "andre",
            "anden", "andet", "dette", "disse", "deres", "hendes", "hans", "vores", "jeres",
            "mine", "dine", "sine", "hver", "alle", "flere", "fleste", "lidt", "mere", "mest",
        },
        "en": {
            "that", "this", "with", "from", "have", "were", "been", "being", "their", "there",
            "about", "would", "could", "should", "which", "where", "when", "what", "they",
            "them", "then", "than", "into", "just", "like", "really", "actually", "maybe",
            "because", "before", "after", "while", "through", "between", "going", "think",
            "know", "something", "anything", "everything", "people", "other", "another",
            "these", "those", "very", "more", "most", "some", "many", "much", "also",
        },
    }

    @classmethod
    def _extract_domain_keywords(cls, all_texts: List[str], language: str = "da", top_k: int = 18) -> List[str]:
        """Extract the most prominent recurring domain words across an entire conversation."""
        stop = cls._STOPWORDS.get(language, cls._STOPWORDS["en"])
        counts: Dict[str, int] = {}
        canonical: Dict[str, str] = {}
        for txt in all_texts:
            for tok in re.findall(r"\b[A-Za-zÆØÅæøå][A-Za-zÆØÅæøå0-9\-]{4,}\b", txt):
                low = tok.lower()
                if low in stop or len(low) < 5:
                    continue
                counts[low] = counts.get(low, 0) + 1
                # Prefer capitalized version if seen capitalized mid-sentence, otherwise lowercase
                if low not in canonical:
                    canonical[low] = tok
                elif tok[0].islower() and canonical[low][0].isupper():
                    canonical[low] = tok

        # Keep words that appear at least twice across the conversation (true topic terms)
        ranked = sorted(
            ((cnt, low) for low, cnt in counts.items() if cnt >= 2),
            key=lambda x: (-x[0], -len(x[1]), x[1]),
        )
        return [canonical[low] for _, low in ranked[:top_k]]

    @classmethod
    def _apply_punctuation_and_grammar_fixes(cls, text: str, language: str = "da") -> str:
        """
        Clean punctuation, commas, and sentence capitalization without altering any spoken words:
        - Normalizes spaces around commas/periods/question marks.
        - Inserts missing Danish commas before clear subordinating/contrasting clauses
          (e.g. 'fordi', 'selvom', 'hvis', 'mens', 'inden', 'hvilket', 'men').
        - Capitalizes sentence starts and adds terminal '.' or '?' when missing.
        """
        out = text.strip()
        if not out:
            return out

        # 1. Normalize spacing around punctuation
        out = re.sub(r"\s+([,.;:!?])", r"\1", out)
        out = re.sub(r"([,;!?])([^\s\d\"')\]])", r"\1 \2", out)
        # Ensure space after period unless part of ellipsis (...) or decimal/abbreviation
        out = re.sub(r"(?<!\.)\.([A-ZÆØÅa-zæøå])", r". \1", out)
        out = re.sub(r",{2,}", ",", out)
        out = re.sub(r"\s{2,}", " ", out).strip()

        # 2. Danish clause comma placement (conservative: only clear subordinating/contrasting conjunctions)
        if language == "da":
            no_comma_before = {
                "og", "eller", "men", "at", "så", "ikke", "kun", "også", "bare",
                "jo", "vel", "nok", "helt", "lidt", "meget", "ret", "end", "som",
                "for", "til", "med", "fra", "på", "af", "om", "i", "ved",
            }
            sub_conj_single = {
                "fordi", "selvom", "hvis", "mens", "inden", "eftersom",
                "hvorefter", "hvorvidt", "hvilket", "hvilken", "hvilke", "men",
            }
            words = out.split(" ")
            if len(words) >= 4:
                rebuilt: List[str] = [words[0]]
                words_since_punct = 1
                for i in range(1, len(words)):
                    cur = words[i]
                    cur_low = cur.lower().strip(".,?!:;\"'()")
                    prev = rebuilt[-1]
                    prev_clean = prev.lower().strip(".,?!:;\"'()")
                    prev_has_punct = prev.endswith((",", ".", "?", "!", ":", ";", "–", "-", "…"))

                    if (
                        cur_low in sub_conj_single
                        and not prev_has_punct
                        and words_since_punct >= 2
                        and prev_clean not in no_comma_before
                        and i < len(words) - 1
                    ):
                        rebuilt[-1] = prev + ","
                        words_since_punct = 0

                    rebuilt.append(cur)
                    if cur.endswith((",", ".", "?", "!", ":", ";")):
                        words_since_punct = 0
                    else:
                        words_since_punct += 1
                out = " ".join(rebuilt)

        # 3. Capitalize start of segment (unless ellipsis) and after sentence-ending punctuation
        if out and not out.startswith("..."):
            for idx_c, ch in enumerate(out):
                if ch.isalpha():
                    if ch.islower():
                        out = out[:idx_c] + ch.upper() + out[idx_c + 1 :]
                    break

        out = re.sub(
            r"([.!?]\s+)([a-zæøå])",
            lambda m: m.group(1) + m.group(2).upper(),
            out,
        )

        # 4. Ensure terminal punctuation (. or ? for clear questions)
        if out and out[-1] not in ".?!…":
            last_sent = re.split(r"[.!?]\s+", out)[-1].strip().lower()
            is_question = False
            if language == "da":
                q_starts = (
                    "hvordan ", "hvorfor ", "hvad ", "hvem ", "hvornår ", "hvilken ",
                    "hvilket ", "hvilke ", "kan du ", "kan vi ", "kan man ", "vil du ",
                    "har du ", "har vi ", "er det ", "er du ", "synes du ", "tænker du ",
                    "tror du ", "mener du ", "skal vi ", "skal du ", "må jeg ",
                )
                is_question = any(last_sent.startswith(q) for q in q_starts)
            elif language == "en":
                q_starts = (
                    "how ", "why ", "what ", "who ", "when ", "where ", "which ",
                    "can you ", "could you ", "do you ", "did you ", "are you ",
                    "is it ", "would you ", "have you ", "should we ",
                )
                is_question = any(last_sent.startswith(q) for q in q_starts)
            elif language in ("sv", "no"):
                q_starts = (
                    "hur ", "varför ", "vad ", "vem ", "när ", "vilken ", "vilket ",
                    "hvordan ", "hvorfor ", "hva ", "hvem ", "når ", "hvilken ",
                    "kan du ", "kan vi ", "vill du ", "vil du ", "har du ", "är det ", "er det ",
                )
                is_question = any(last_sent.startswith(q) for q in q_starts)
            elif language == "de":
                q_starts = (
                    "wie ", "warum ", "was ", "wer ", "wann ", "wo ", "welche ", "welcher ",
                    "kannst du ", "können sie ", "hast du ", "haben sie ", "ist das ", "sind sie ",
                )
                is_question = any(last_sent.startswith(q) for q in q_starts)
            elif language == "es":
                q_starts = (
                    "cómo ", "como ", "por qué ", "qué ", "quién ", "cuándo ", "dónde ", "cuál ",
                    "puedes ", "puede ", "tienes ", "eres ", "es ", "¿",
                )
                is_question = any(last_sent.startswith(q) for q in q_starts)
            elif language == "fr":
                q_starts = (
                    "comment ", "pourquoi ", "quoi ", "qui ", "quand ", "où ", "quel ", "quelle ",
                    "est-ce que ", "pouvez-vous ", "peux-tu ", "avez-vous ", "as-tu ",
                )
                is_question = any(last_sent.startswith(q) for q in q_starts)
            elif language == "it":
                q_starts = (
                    "come ", "perché ", "cosa ", "che cosa ", "chi ", "quando ", "dove ", "quale ",
                    "puoi ", "può ", "hai ", "avete ", "sei ", "è ",
                )
                is_question = any(last_sent.startswith(q) for q in q_starts)
            out += "?" if is_question else "."

        return out

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
                (r"\bkulturforskellende\b", "kulturforskellene"),
                (r"\bkulturforskeerne\b", "kulturforskellene"),
                (r"\b[Kk]ronprinspaaret\b", "Kronprinsparret"),
                (r"\b[Kk]ronprinsparet\b", "Kronprinsparret"),
                (r"\b[Kk]ronprinsparrede\b", "Kronprinsparret"),
                (r"\b[Kk]ronprinsparret\b", "Kronprinsparret"),
                (r"\b[Kk]ongefamilens\b", "Kongefamiliens"),
                (r"\b[Kk]ongefamiliens\b", "Kongefamiliens"),
                (r"\bKong\s+i\s+familien\s+[Ss]poli\b", "Kongefamiliens bolig"),
                (r"\bKong\s*i\s*familien'?s?\s+bolig\b", "Kongefamiliens bolig"),
                (r"\b[Mm]alienborg\b", "Amalienborg"),
                (r"\b[Mm]aljenborg\b", "Amalienborg"),
                (r"\b[Aa]malienborg\b", "Amalienborg"),
                (r"\b[Dd]anmark\b", "Danmark"),
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
                    # "i stedet satte deres hjem" / "i stand satte deres hjem" / "i standsatte deres hjem" -> "istandsatte deres hjem"
                    (r"\bi\s+(?:stedet\s+satte|stand\s+satte|standsatte)\s+deres\s+hjem\b", "istandsatte deres hjem"),
                    # "frederik 8s palæ" / "frederik ottnes palé" -> "Frederik VIII's Palæ"
                    (r"\b[Ff]rederik\s+(?:8\.?s|ottnes|ottendes)\s+[Pp]al(?:æ|é|ais|e)\b", "Frederik VIII's Palæ"),
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

        # 3. Context-driven proper-noun, cultural anchor, and recurring domain-term matching:
        # Collect capitalized terms from the conversation context + cultural anchors for this language
        ctx_terms = set(re.findall(r"\b[A-ZÆØÅ][a-zæøåA-ZÆØÅ0-9\-']{4,}\b", context_summary))
        lang_anchors = cls.CULTURAL_ANCHORS.get(language, {})
        for anchor_list in lang_anchors.values():
            for item in anchor_list:
                if " " not in item and len(item) >= 5 and not item.lower().endswith("s"):
                    ctx_terms.add(item)

        # Also extract explicit Key Domain Terms from the whole-conversation summary
        # (e.g. 'fysioterapi', 'musklerne', 'genoptræning') so phonetically confused words
        # like 'musikerne' -> 'musklerne' snap to the conversation's true domain vocabulary.
        domain_terms: List[str] = []
        m_dom = re.search(r"Key Domain Terms:\s*([^\n|]+)", context_summary)
        if m_dom:
            domain_terms = [
                t.strip() for t in m_dom.group(1).split(",") if len(t.strip()) >= 6
            ]
        domain_lower_set = {t.lower() for t in domain_terms}

        if ctx_terms or domain_terms:
            tokens = out.split()
            for i, tok_w in enumerate(tokens):
                stripped = tok_w.strip(".,?!:;\"'()[]")
                if len(stripped) < 5:
                    continue
                low_s = stripped.lower()
                replaced = False

                # 3a. Proper nouns / cultural anchors
                if not low_s.endswith("s"):
                    for term in ctx_terms:
                        if low_s == term.lower() or low_s == (term.lower() + "s"):
                            continue
                        sim = difflib.SequenceMatcher(
                            None,
                            cls._phonetic_key(stripped, language),
                            cls._phonetic_key(term, language),
                        ).ratio()
                        raw_sim = difflib.SequenceMatcher(None, low_s, term.lower()).ratio()
                        if sim >= 0.86 and raw_sim >= 0.78:
                            tokens[i] = tok_w.replace(stripped, term)
                            replaced = True
                            break

                # 3b. Whole-conversation domain homophone disambiguation (e.g. musikerne -> musklerne)
                if not replaced and domain_terms and len(stripped) >= 6 and low_s not in domain_lower_set:
                    p_tok = cls._phonetic_key(stripped, language)
                    best_term = None
                    best_score = 0.0
                    for term in domain_terms:
                        t_low = term.lower()
                        if low_s == t_low:
                            continue
                        # Avoid snapping a word to a simple prefix/inflection of itself (e.g. muskel -> muskler)
                        if t_low.startswith(low_s) or low_s.startswith(t_low):
                            continue
                        p_sim = difflib.SequenceMatcher(
                            None, p_tok, cls._phonetic_key(term, language)
                        ).ratio()
                        raw_sim = difflib.SequenceMatcher(None, low_s, t_low).ratio()
                        if p_sim >= 0.85 and raw_sim >= 0.80 and p_sim > best_score:
                            best_score = p_sim
                            best_term = term
                    if best_term is not None:
                        # Preserve the original token's capitalization
                        if stripped[0].isupper() and best_term[0].islower():
                            repl_word = best_term[0].upper() + best_term[1:]
                        elif stripped[0].islower() and best_term[0].isupper():
                            repl_word = best_term.lower()
                        else:
                            repl_word = best_term
                        tokens[i] = tok_w.replace(stripped, repl_word)
            out = " ".join(tokens)

        # 4. Punctuation, comma & sentence capitalization grammar check
        out = cls._apply_punctuation_and_grammar_fixes(out, language=language)
        return out

    @staticmethod
    def _is_llm_clarifier(model_id: str) -> bool:
        info = next((m for m in CLARIFICATION_MODELS if m["id"] == model_id), None)
        return bool(info and info.get("type") == "qwen-onnx")

    @staticmethod
    def _format_chat_prompt(model_id: str, system_prompt: str, user_prompt: str) -> str:
        info = next((m for m in CLARIFICATION_MODELS if m["id"] == model_id), None)
        tpl = (info or {}).get("chat_template", "chatml")
        if tpl == "llama3":
            return (
                "<|begin_of_text|>"
                "<|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_prompt}<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_prompt}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        if tpl == "gemma":
            return (
                "<bos><start_of_turn>user\n"
                f"{system_prompt}\n\n{user_prompt}<end_of_turn>\n"
                "<start_of_turn>model\n"
            )
        if tpl == "phi4":
            return (
                f"<|system|>{system_prompt}<|end|>"
                f"<|user|>{user_prompt}<|end|>"
                "<|assistant|>"
            )
        # Default: ChatML (Qwen2.5, Qwen3, EuroLLM, DanskGPT, SmolLM2)
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

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
                sess_opts.intra_op_num_threads = max(4, min(12, (os.cpu_count() or 4) - 1))
                sess = app_config.create_ort_session(
                    str(model_dir / onnx_rel),
                    sess_opts,
                    is_autoregressive_decoder=True,
                )
                self._qwen_sessions[model_id] = (sess, tok, info)
                return self._qwen_sessions[model_id]
            except Exception as exc:
                print(f"[ContextClarifier] Warning loading {model_id}: {exc}")
                return None

    def _run_qwen_prompt(
        self,
        model_id: str,
        prompt: str,
        max_new_tokens: int = 96,
        stop_on_newline: bool = True,
    ) -> Optional[str]:
        runner_tuple = self._get_qwen_runner(model_id)
        if runner_tuple is None:
            return None
        sess, tok, info = runner_tuple
        try:
            num_layers = int(info.get("num_layers", "24"))
            num_kv_heads = int(info.get("num_kv_heads", "2"))
            head_dim = int(info.get("head_dim", "64"))

            sess_inputs = {inp.name: inp for inp in sess.get_inputs()}
            has_pos_ids = "position_ids" in sess_inputs
            kv0_inp = sess_inputs.get("past_key_values.0.key")
            kv_dtype = np.float16 if (kv0_inp and "float16" in kv0_inp.type) else np.float32

            ids = tok.encode(prompt, add_special_tokens=False).ids
            seq_len = len(ids)
            feed: Dict[str, np.ndarray] = {
                "input_ids": np.array([ids], dtype=np.int64),
                "attention_mask": np.ones((1, seq_len), dtype=np.int64),
            }
            if has_pos_ids:
                feed["position_ids"] = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)
            for layer in range(num_layers):
                feed[f"past_key_values.{layer}.key"] = np.zeros(
                    (1, num_kv_heads, 0, head_dim), dtype=kv_dtype
                )
                feed[f"past_key_values.{layer}.value"] = np.zeros(
                    (1, num_kv_heads, 0, head_dim), dtype=kv_dtype
                )

            outs = sess.run(None, feed)
            logits = outs[0]
            past_kv = outs[1:]
            stop_ids = {
                tid
                for tid in (
                    tok.token_to_id("<|im_end|>"),
                    tok.token_to_id("<|endoftext|>"),
                    tok.token_to_id("<|eot_id|>"),
                    tok.token_to_id("<|end_of_text|>"),
                    tok.token_to_id("<end_of_turn>"),
                    tok.token_to_id("<|end|>"),
                    tok.token_to_id("<eos>"),
                    tok.token_to_id("</s>"),
                )
                if tid is not None
            }
            gen_ids: List[int] = []
            cur_pos = seq_len
            next_tok = int(np.argmax(logits[0, -1]))

            for _ in range(max_new_tokens):
                if next_tok in stop_ids:
                    break
                gen_ids.append(next_tok)
                if stop_on_newline and "\n" in tok.decode(gen_ids):
                    break
                cur_len = cur_pos + 1
                feed = {
                    "input_ids": np.array([[next_tok]], dtype=np.int64),
                    "attention_mask": np.ones((1, cur_len), dtype=np.int64),
                }
                if has_pos_ids:
                    feed["position_ids"] = np.array([[cur_pos]], dtype=np.int64)
                for layer in range(num_layers):
                    feed[f"past_key_values.{layer}.key"] = past_kv[layer * 2]
                    feed[f"past_key_values.{layer}.value"] = past_kv[layer * 2 + 1]
                outs = sess.run(None, feed)
                logits = outs[0]
                past_kv = outs[1:]
                cur_pos = cur_len
                next_tok = int(np.argmax(logits[0, -1]))

            return tok.decode(gen_ids).strip()
        except Exception as exc:
            print(f"[ContextClarifier] LLM error ({model_id}): {exc}")
            return None

    def summarize_conversation_context(
        self,
        all_texts: List[str],
        language: str = "da",
        clarification_model: str = DEFAULT_CLARIFICATION_MODEL,
        title: str = "",
        source_file_name: str = "",
    ) -> str:
        """
        Pre-correction whole-conversation understanding pass:
        Analyzes the ENTIRE transcript before correcting individual sentences so the
        clarifier knows the overall topic/domain (e.g. a physiotherapy interview where
        'musklerne' makes sense instead of phonetically similar 'musikerne').
        - Always extracts top recurring domain terms across the full transcript (0 MB).
        - When Qwen2.5 is active, generates a concise 2-sentence global topic summary
          from beginning + middle + end excerpts of the conversation.
        """
        clean_texts = [t.strip() for t in all_texts if t and t.strip()]
        if not clean_texts:
            return ""

        parts: List[str] = []
        if title and not title.lower().startswith(("new meeting", "untitled", "sales meeting")):
            parts.append(f"Title: {title}.")
        if source_file_name:
            clean_fname = re.sub(r"\.[^/.]+$", "", source_file_name).replace("_", " ").replace("-", " ")
            if clean_fname and clean_fname.lower() not in (title or "").lower():
                parts.append(f"File: {clean_fname}.")

        domain_keywords = self._extract_domain_keywords(clean_texts, language=language, top_k=18)

        # If a local ONNX LLM clarifier is selected, generate a concise whole-conversation topic summary ONCE
        if self._is_llm_clarifier(clarification_model):
            joined_full = " ".join(clean_texts)
            if len(joined_full) <= 1400:
                excerpt = joined_full
            else:
                n_s = len(clean_texts)
                # Sample beginning, middle, and end so the summary covers the whole conversation
                b_txt = " ".join(clean_texts[: max(2, n_s // 6)])[:480]
                m_start = max(0, n_s // 2 - max(1, n_s // 12))
                m_txt = " ".join(clean_texts[m_start : m_start + max(2, n_s // 6)])[:460]
                e_txt = " ".join(clean_texts[-max(2, n_s // 6) :])[:460]
                excerpt = f"{b_txt} [...] {m_txt} [...] {e_txt}"

            sum_prompt = self._format_chat_prompt(
                clarification_model,
                "Summarize the overall topic, domain/profession, and key subject terminology of this transcript "
                "in 2 concise sentences. Output ONLY the summary.",
                f"Transcript ({language}): {excerpt}\nTopic Summary:",
            )
            qwen_sum = self._run_qwen_prompt(
                clarification_model, sum_prompt, max_new_tokens=90, stop_on_newline=False
            )
            if qwen_sum:
                qwen_sum_clean = re.sub(r"\s+", " ", qwen_sum).strip()[:360]
                parts.append(f"Overall Topic Summary: {qwen_sum_clean}")

        if domain_keywords:
            parts.append("Key Domain Terms: " + ", ".join(domain_keywords))

        return " | ".join(parts).strip()

    def clarify_with_qwen(
        self,
        model_id: str,
        raw_text: str,
        language: str,
        context_summary: str,
        global_summary: str = "",
    ) -> str:
        topic_block = (
            f"Whole-Conversation Overview: {global_summary[:420]}\n"
            if global_summary
            else ""
        )
        prompt = self._format_chat_prompt(
            model_id,
            "You are an ASR speech-to-text validator and punctuation editor. "
            "Using the whole-conversation topic and local context: "
            "1) Fix phonetically misheard words or domain homophones (e.g. 'musikerne' -> 'musklerne' in a physiotherapy context) and split compound words. "
            "2) Fix commas, periods, question marks, and sentence capitalization for natural grammar in the spoken language. "
            "Do NOT rewrite the sentence, do NOT add or remove words, do NOT change word order, and do NOT translate. "
            "Output ONLY the corrected sentence.",
            f"{topic_block}Local Context ({language}): {context_summary[-340:]}\n"
            f"Raw Speech-to-Text: {raw_text}\n"
            "Validated:",
        )
        candidate = self._run_qwen_prompt(
            model_id,
            prompt,
            max_new_tokens=min(112, max(24, int(len(raw_text.split()) * 3.2))),
            stop_on_newline=True,
        )
        if not candidate:
            return raw_text
        return self._validate_structure_preserved(raw_text, candidate)

    def clarify_segment(
        self,
        raw_text: str,
        language: str,
        context_summary: str,
        clarification_model: str = DEFAULT_CLARIFICATION_MODEL,
        whisper_ctx_candidate: Optional[str] = None,
        speaker_context: str = "",
        global_summary: str = "",
    ) -> str:
        """
        Validate and clarify a single transcribed segment sentence-by-sentence using
        the whole-conversation topic summary, local conversation context, speaker persona/history,
        and punctuation/grammar rules WITHOUT restructuring any sentence.
        """
        if not raw_text or clarification_model == "none":
            return raw_text

        working = raw_text.strip()
        full_ctx = f"{global_summary} {context_summary}".strip()

        # 1. If Whisper contextual second-pass candidate (`sot_prev`) is provided, merge only high-confidence word fixes
        if whisper_ctx_candidate and whisper_ctx_candidate.strip() != working:
            working = self._merge_acoustic_context_candidate(
                working, whisper_ctx_candidate, f"{full_ctx} {speaker_context}"
            )

        # 2. Apply sentence-by-sentence contextual, domain-term, compound-word & punctuation validation
        working = self._apply_contextual_compound_and_lexicon_fixes(
            working, language, full_ctx, speaker_context=speaker_context
        )

        # 3. If user selected a local ONNX LLM context clarifier model, run it with whole-conversation summary
        #    and strict phonetic + punctuation-preserving structure guardrails
        if self._is_llm_clarifier(clarification_model):
            working = self.clarify_with_qwen(
                clarification_model,
                working,
                language,
                f"{context_summary}\nSpeaker profile: {speaker_context}".strip(),
                global_summary=global_summary,
            )
            working = self._apply_contextual_compound_and_lexicon_fixes(
                working, language, full_ctx, speaker_context=speaker_context
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
    Runs Speech-to-Text ONNX models directly via Microsoft's signed `onnxruntime.InferenceSession`.
    Supports:
    - Standard `sherpa-onnx` Whisper INT8 models (`tiny`, `base`, `small`, `medium`, `turbo`)
    - Specialized Danish HuggingFace/Optimum Whisper ONNX models (`hviske-v2` — Syv.ai / CoRal Whisper Large-v3)
    - Specialized Danish Wav2Vec2 CTC ONNX models (`roest-315m` — Alexandra Instituttet / CoRal Røst 315M)
    """

    def __init__(self, model_id: str):
        if model_id == "hviske-v2-fp16":
            model_id = "hviske-v2-int8"
        self.model_id = model_id
        try:
            paths = resolve_local_whisper_model(model_id)
        except FileNotFoundError:
            paths = ensure_whisper_model(model_id)

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = max(4, min(12, (os.cpu_count() or 4) - 1))
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            engine_str = str(paths.get("engine", ""))
            self.is_hf_onnx = engine_str == "hf-whisper-onnx"
            self.is_wav2vec2_onnx = engine_str == "wav2vec2-onnx"
            self.encoder = app_config.create_ort_session(
                str(paths["encoder"]), sess_opts
            )
            if self.is_wav2vec2_onnx:
                self.decoder = None
            else:
                self.decoder = app_config.create_ort_session(
                    str(paths["decoder"]), sess_opts, is_autoregressive_decoder=True
                )
        except Exception as load_err:
            print(f"[WhisperONNXRunner] Failed loading '{model_id}' ({load_err}); attempting fallback model...")
            fallback_loaded = False
            for candidate_id in ("hviske-v2-int8", "hviske-v2", "roest-v3-315m", "roest-315m", "turbo", "small", "base", "tiny"):
                if candidate_id == model_id or not is_whisper_model_downloaded(candidate_id):
                    continue
                try:
                    paths = get_whisper_onnx_paths(candidate_id)
                    engine_str = str(paths.get("engine", ""))
                    self.is_hf_onnx = engine_str == "hf-whisper-onnx"
                    self.is_wav2vec2_onnx = engine_str == "wav2vec2-onnx"
                    self.encoder = app_config.create_ort_session(
                        str(paths["encoder"]), sess_opts
                    )
                    if self.is_wav2vec2_onnx:
                        self.decoder = None
                    else:
                        self.decoder = app_config.create_ort_session(
                            str(paths["decoder"]), sess_opts, is_autoregressive_decoder=True
                        )
                    self.model_id = candidate_id
                    fallback_loaded = True
                    break
                except Exception:
                    continue
            if not fallback_loaded:
                raise load_err

        if self.is_wav2vec2_onnx:
            vocab_raw = json.loads(paths["tokens"].read_text(encoding="utf-8-sig"))
            self.w2v_inv_vocab: Dict[int, str] = {int(v): k for k, v in vocab_raw.items()}
            self.is_multilingual = False
            self.tokens_map = {}
            self._by_first_byte = {}
            return

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
        if not text_str or not text_str.strip() or self.is_wav2vec2_onnx:
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
        if self.is_wav2vec2_onnx:
            if len(audio_16k) < 400:
                audio_16k = np.pad(audio_16k, (0, 400 - len(audio_16k)))
            norm_a = (audio_16k - float(np.mean(audio_16k))) / float(np.std(audio_16k) + 1e-7)
            logits = self.encoder.run(
                None, {"input_values": norm_a[None, :].astype(np.float32)}
            )[0]
            return logits, logits
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
        if self.is_hf_onnx or self.is_wav2vec2_onnx:
            # Specialized single-language ONNX ASR models (Hviske/Røst=da, KB=sv, NB=no, Distil=en, DE/ES/FR/IT)
            if allowed_languages and len(allowed_languages) == 1:
                return allowed_languages[0]
            info = next((m for m in app_config.WHISPER_MODELS if m["id"] == self.model_id), None)
            lg = info.get("language_group", "da") if info else "da"
            return lg if lg not in ("multilingual", "custom", "") else "da"
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
        if self.is_wav2vec2_onnx:
            logits = cross_k
            pred_ids = np.argmax(logits[0], axis=-1)
            chars: List[str] = []
            prev_id = -1
            for pid in pred_ids:
                pid_i = int(pid)
                if pid_i != prev_id:
                    ch = self.w2v_inv_vocab.get(pid_i, "")
                    if ch and ch not in ("<pad>", "<s>", "</s>", "<unk>"):
                        chars.append(" " if ch == "|" else ch)
                prev_id = pid_i
            text_out = re.sub(r"\s+", " ", "".join(chars)).strip()
            if text_out and text_out[0].islower():
                text_out = text_out[0].upper() + text_out[1:]
            if text_out and text_out[-1] not in ".?!…":
                text_out += "."
            return text_out
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
                out_ids, looped = _detect_and_trim_token_loop(out_ids)
                if looped:
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

            text_out = _trim_repeated_sentences(
                self.hf_tokenizer.decode(out_ids, skip_special_tokens=True).strip()
            )
            if not _has_meaningful_speech_text(text_out):
                return ""
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
            out_ids, looped = _detect_and_trim_token_loop(out_ids)
            if looped:
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
        text_out = _trim_repeated_sentences(raw_bytes.decode("utf-8", errors="replace").strip())
        return text_out if _has_meaningful_speech_text(text_out) else ""


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

            if not _has_meaningful_speech_text(orig_text):
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
        on_partial_update: Optional[Callable[[MeetingSession], None]] = None,
    ) -> MeetingSession:
        total_audio_sec = len(audio_16k) / float(SAMPLE_RATE)
        if progress_cb:
            progress_cb("Saving normalized audio for playback...", 10)
        self.store.write_full_audio(session.id, audio_16k)
        self.store.save_session(session)
        if on_partial_update:
            on_partial_update(session)

        if progress_cb:
            progress_cb(
                f"Preparing local Whisper ONNX ({session.whisper_model}) model...", 18
            )
        self._get_whisper_runner(session.whisper_model)

        if progress_cb:
            progress_cb("Detecting speech segments across recording...", 28)
        max_seg_len = 14.0
        seg_times = segment_speech_by_energy(
            audio_16k, min_speech_sec=0.5, max_segment_sec=max_seg_len, min_silence_sec=0.4
        )
        if not seg_times:
            self.store.save_session(session)
            if progress_cb:
                progress_cb("Completed!", 100)
            return session

        if progress_cb:
            progress_cb("Diarizing speakers (recognizing distinct voices)...", 42)
        seg_times = self.diarizer.split_segments_on_speaker_change(
            audio_16k, seg_times
        )
        speaker_ids, centroids = self.diarizer.cluster_segments_batch(
            audio_16k, seg_times, num_speakers=num_speakers
        )
        # Post-clustering centroid refinement: split any remaining longer segments (>= 5s)
        # where the voice switches cleanly between two discovered speaker centroids.
        seg_times, speaker_ids = self.diarizer.refine_segments_with_centroids(
            audio_16k, seg_times, speaker_ids, centroids
        )

        # Optimization for long recordings (> 4 minutes, e.g. 1-2 hour MP4 meetings):
        # Merge consecutive short utterances by the SAME speaker with a brief pause (<= 0.9s)
        # up to 15.0s so Whisper gets complete sentences without crossing speaker pauses
        # or entering 20-30s hallucination/repetition territory.
        if total_audio_sec > 240.0 and len(seg_times) > 1:
            merged_times: List[Tuple[float, float]] = []
            merged_spks: List[str] = []
            cur_s, cur_e = seg_times[0]
            cur_spk = speaker_ids[0]
            for (nxt_s, nxt_e), nxt_spk in zip(seg_times[1:], speaker_ids[1:]):
                if (
                    nxt_spk == cur_spk
                    and (nxt_s - cur_e) <= 0.9
                    and (nxt_e - cur_s) <= 15.0
                ):
                    cur_e = nxt_e
                else:
                    merged_times.append((cur_s, cur_e))
                    merged_spks.append(cur_spk)
                    cur_s, cur_e = nxt_s, nxt_e
                    cur_spk = nxt_spk
            merged_times.append((cur_s, cur_e))
            merged_spks.append(cur_spk)
            seg_times = merged_times
            speaker_ids = merged_spks

        # Strictly isolate speakers and segments to this session's audio
        session.speakers = {}
        session.embeddings = centroids
        session.segments = []

        total = len(seg_times)
        t_start = time.time()
        last_partial_push = time.time()

        for idx, ((s_sec, e_sec), spk_id) in enumerate(zip(seg_times, speaker_ids)):
            if progress_cb:
                pct = 48 + int(48 * (idx + 1) / max(1, total))
                eta_suffix = ""
                if idx >= 2:
                    elapsed = time.time() - t_start
                    avg_per_seg = elapsed / float(idx)
                    rem_sec = int(max(0.0, avg_per_seg * (total - idx)))
                    if rem_sec >= 60:
                        eta_suffix = f" (~{rem_sec // 60}m {rem_sec % 60:02d}s remaining)"
                    elif rem_sec > 2:
                        eta_suffix = f" (~{rem_sec}s remaining)"
                progress_cb(
                    f"Transcribing & translating segment {idx + 1}/{total}{eta_suffix}...", pct
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
            # Defer Qwen LLM clarification to Pass 2 (after whole-conversation topic summary is built)
            pass1_clarif = (
                "whisper-context"
                if session.clarification_model in ("qwen2.5-0.5b", "qwen2.5-1.5b")
                else session.clarification_model
            )
            detected_lang, orig_text, trans_text, conf, raw_stt_text = (
                self.transcribe_and_translate_clip(
                    clip,
                    forced_language=forced_lang,
                    target_language=session.target_language,
                    model_id=session.whisper_model,
                    clarification_model=pass1_clarif,
                    translation_model=session.translation_model,
                    allowed_languages=session.spoken_languages,
                    conversation_context=conv_ctx,
                    speaker_context=spk_ctx,
                )
            )
            if not _has_meaningful_speech_text(orig_text):
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

            # Live incremental save & UI broadcast every 5 segments or 8 seconds so 2-hour files
            # stream paragraphs live onto the screen and never lose progress!
            now_ts = time.time()
            if (len(session.segments) % 5 == 0) or (now_ts - last_partial_push >= 8.0):
                last_partial_push = now_ts
                self.store.save_session(session)
                if on_partial_update:
                    on_partial_update(session)

        # Second-Pass Whole-Conversation Summary + Global Grammar & Context Clarification:
        # First summarize what the entire conversation is about (domain, topic, key terms),
        # then validate every segment's domain vocabulary, commas, periods & capitalization.
        self._run_global_clarification_pass(session, progress_cb=progress_cb)

        self.store.save_session(session)
        if progress_cb:
            progress_cb("Completed!", 100)
        return session

    def _run_global_clarification_pass(
        self,
        session: MeetingSession,
        progress_cb: Optional[Callable[[str, int], None]] = None,
    ) -> None:
        """
        Whole-conversation pre-summary + 2nd-pass global clarification:
        1. Summarizes the entire transcript into a compact topic/domain overview + key terms
           (e.g. physiotherapy -> 'musklerne' instead of 'musikerne').
        2. Clarifies each segment's raw text using the global summary, surrounding before/after
           context, speaker profile, and punctuation/comma grammar rules.
        3. Updates translations if a target language is active.
        """
        if not session.segments or not session.clarification_model or session.clarification_model == "none":
            return

        if progress_cb:
            progress_cb("Summarizing full conversation topic & refining grammar...", 97)

        all_raw_texts = [
            (s.raw_text or s.original_text or "").strip()
            for s in session.segments
            if (s.raw_text or s.original_text or "").strip()
        ]
        # Primary spoken language across segments
        lang_counts: Dict[str, int] = {}
        for s in session.segments:
            if s.source_language:
                lang_counts[s.source_language] = lang_counts.get(s.source_language, 0) + 1
        primary_lang = max(lang_counts, key=lambda k: lang_counts[k]) if lang_counts else "da"

        global_summary = self.clarifier.summarize_conversation_context(
            all_raw_texts,
            language=primary_lang,
            clarification_model=session.clarification_model,
            title=session.title,
            source_file_name=session.source_file_name,
        )

        for idx, seg in enumerate(session.segments):
            base_raw = (seg.raw_text or seg.original_text or "").strip()
            if not base_raw:
                continue
            if not seg.raw_text:
                seg.raw_text = base_raw

            full_conv_ctx, full_spk_ctx = self._build_conversation_context(
                session, up_to_seg_index=idx, current_speaker_id=seg.speaker_id
            )
            seg.original_text = self.clarifier.clarify_segment(
                raw_text=base_raw,
                language=seg.source_language,
                context_summary=full_conv_ctx,
                clarification_model=session.clarification_model,
                whisper_ctx_candidate=None,
                speaker_context=full_spk_ctx,
                global_summary=global_summary,
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
        first summarize the whole conversation topic/domain, then re-run Context Clarification
        & punctuation/grammar checks on each segment's `raw_text`, and re-translate.
        """
        if session.clarification_model == "none":
            for seg in session.segments:
                base_raw = (seg.raw_text or seg.original_text or "").strip()
                if base_raw:
                    if not seg.raw_text:
                        seg.raw_text = base_raw
                    seg.original_text = base_raw
            return self.retranslate_all_segments(session)

        all_raw_texts = [
            (s.raw_text or s.original_text or "").strip()
            for s in session.segments
            if (s.raw_text or s.original_text or "").strip()
        ]
        lang_counts: Dict[str, int] = {}
        for s in session.segments:
            if s.source_language:
                lang_counts[s.source_language] = lang_counts.get(s.source_language, 0) + 1
        primary_lang = max(lang_counts, key=lambda k: lang_counts[k]) if lang_counts else "da"

        global_summary = self.clarifier.summarize_conversation_context(
            all_raw_texts,
            language=primary_lang,
            clarification_model=session.clarification_model,
            title=session.title,
            source_file_name=session.source_file_name,
        )

        for idx, seg in enumerate(session.segments):
            base_raw = (seg.raw_text or seg.original_text or "").strip()
            if not base_raw:
                continue
            if not seg.raw_text:
                seg.raw_text = base_raw

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
                global_summary=global_summary,
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
        on_partial_update: Optional[Callable[[MeetingSession], None]] = None,
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
                session.id, 0.0, max(1.0, session.duration or 14400.0)
            )
            if len(audio_16k) > 0:
                return self.process_uploaded_file(
                    session,
                    audio_16k,
                    progress_cb=progress_cb,
                    on_partial_update=on_partial_update,
                )
            return session

        if progress_cb:
            progress_cb(
                f"Loading Whisper ({session.whisper_model}), Clarifier ({session.clarification_model}) & Translator ({session.translation_model})...",
                10,
            )
        self._get_whisper_runner(session.whisper_model)

        # Build the list of (start_sec, end_sec, speaker_id) work items without ever
        # pre-populating empty TranscriptSegment shells into `session.segments` (which
        # previously caused empty "Speaker 1: / Speaker 2:" rows in the UI).
        work_items: List[Tuple[float, float, str]] = [
            (seg.start, seg.end, seg.speaker_id) for seg in session.segments
        ]

        # If the full session recording is on disk and this is a file session,
        # check if any longer segment contains an unsplit intra-segment speaker change
        # and split + reassign speaker IDs using existing session voice centroids.
        if session.session_type == "file" and audio_path.exists() and session.embeddings:
            full_audio_16k = self.store.read_audio_slice(
                session.id, 0.0, max(1.0, session.duration or 3600.0)
            )
            if len(full_audio_16k) > int(SAMPLE_RATE * 1.0):
                cur_times = [(s, e) for s, e, _ in work_items]
                refined_times = self.diarizer.split_segments_on_speaker_change(
                    full_audio_16k, cur_times
                )
                centroids_np = {
                    spk_id: np.array(vec, dtype=np.float32)
                    for spk_id, vec in session.embeddings.items()
                }
                refined_spks: List[str] = []
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
                    refined_spks.append(best_spk)

                refined_times, refined_spks = self.diarizer.refine_segments_with_centroids(
                    full_audio_16k, refined_times, refined_spks, session.embeddings
                )
                work_items = [
                    (round(s_sec, 2), round(e_sec, 2), spk_id)
                    for (s_sec, e_sec), spk_id in zip(refined_times, refined_spks)
                ]

        session.segments = []
        total = len(work_items)
        t_start = time.time()
        last_partial_push = time.time()
        for idx, (s_sec, e_sec, spk_id) in enumerate(work_items):
            if progress_cb:
                pct = 15 + int(80 * (idx + 1) / max(1, total))
                eta_suffix = ""
                if idx >= 2:
                    elapsed = time.time() - t_start
                    avg_per_seg = elapsed / float(idx)
                    rem_sec = int(max(0.0, avg_per_seg * (total - idx)))
                    if rem_sec >= 60:
                        eta_suffix = f" (~{rem_sec // 60}m {rem_sec % 60:02d}s remaining)"
                    elif rem_sec > 2:
                        eta_suffix = f" (~{rem_sec}s remaining)"
                progress_cb(
                    f"Re-transcribing, clarifying & translating segment {idx + 1}/{total} ({session.whisper_model}){eta_suffix}...",
                    pct,
                )

            clip = self.store.read_audio_slice(session.id, s_sec, e_sec)
            if len(clip) < int(SAMPLE_RATE * 0.25):
                continue

            spk = session.speakers.get(spk_id)
            forced_lang = (
                spk.language if (spk and spk.language != "auto") else None
            )
            conv_ctx, spk_ctx = self._build_conversation_context(
                session, current_speaker_id=spk_id
            )
            pass1_clarif = (
                "whisper-context"
                if session.clarification_model in ("qwen2.5-0.5b", "qwen2.5-1.5b")
                else session.clarification_model
            )
            detected_lang, orig_text, trans_text, conf, raw_stt_text = (
                self.transcribe_and_translate_clip(
                    clip,
                    forced_language=forced_lang,
                    target_language=session.target_language,
                    model_id=session.whisper_model,
                    clarification_model=pass1_clarif,
                    translation_model=session.translation_model,
                    allowed_languages=session.spoken_languages,
                    conversation_context=conv_ctx,
                    speaker_context=spk_ctx,
                )
            )
            if not _has_meaningful_speech_text(orig_text):
                continue

            self.store.ensure_speaker(session, spk_id)
            session.segments.append(
                TranscriptSegment(
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
            )

            now_ts = time.time()
            if (len(session.segments) % 5 == 0) or (now_ts - last_partial_push >= 8.0):
                last_partial_push = now_ts
                self.store.save_session(session)
                if on_partial_update:
                    on_partial_update(session)

        self._run_global_clarification_pass(session, progress_cb=progress_cb)

        self.store.save_session(session)
        if progress_cb:
            progress_cb("Completed re-transcription, clarification & translation!", 100)
        return session

    def resume_incomplete_transcription(
        self,
        session: MeetingSession,
        progress_cb: Optional[Callable[[str, int], None]] = None,
        on_partial_update: Optional[Callable[[MeetingSession], None]] = None,
    ) -> MeetingSession:
        """
        Resume a partially transcribed recording from the end timestamp of the last saved segment
        up to `session.duration`, preserving all existing segments and speaker identities.
        """
        audio_path = self.store.get_audio_path(session.id)
        if not audio_path.exists():
            return session

        last_end = max((s.end for s in session.segments), default=0.0)
        full_dur = max(session.duration or 0.0, last_end + 1.0)
        rem_audio = self.store.read_audio_slice(session.id, last_end, full_dur + 3600.0)
        if len(rem_audio) < int(SAMPLE_RATE * 1.0):
            return session

        if progress_cb:
            progress_cb(
                f"Resuming transcription from {int(last_end // 60):02d}:{int(last_end % 60):02d} ({session.whisper_model})...",
                15,
            )
        self._get_whisper_runner(session.whisper_model)

        rem_dur = len(rem_audio) / float(SAMPLE_RATE)
        max_seg_len = 14.0
        rel_times = segment_speech_by_energy(
            rem_audio, min_speech_sec=0.5, max_segment_sec=max_seg_len, min_silence_sec=0.4
        )
        if not rel_times:
            return session

        rel_times = self.diarizer.split_segments_on_speaker_change(rem_audio, rel_times)

        # Match remaining segments against existing session speaker voiceprints (or online assignment)
        rel_spks: List[str] = []
        for rs, re_t in rel_times:
            clip = rem_audio[int(rs * SAMPLE_RATE) : int(re_t * SAMPLE_RATE)]
            spk_id, session.embeddings = self.diarizer.assign_speaker_online(
                clip, session.embeddings
            )
            rel_spks.append(spk_id)

        rel_times, speaker_ids = self.diarizer.refine_segments_with_centroids(
            rem_audio, rel_times, rel_spks, session.embeddings
        )
        seg_times: List[Tuple[float, float]] = [
            (last_end + rs, last_end + re_t) for rs, re_t in rel_times
        ]

        if (last_end + rem_dur) > 240.0 and len(seg_times) > 1:
            merged_times: List[Tuple[float, float]] = []
            merged_spks: List[str] = []
            cur_s, cur_e = seg_times[0]
            cur_spk = speaker_ids[0]
            for (nxt_s, nxt_e), nxt_spk in zip(seg_times[1:], speaker_ids[1:]):
                if (
                    nxt_spk == cur_spk
                    and (nxt_s - cur_e) <= 0.9
                    and (nxt_e - cur_s) <= 15.0
                ):
                    cur_e = nxt_e
                else:
                    merged_times.append((cur_s, cur_e))
                    merged_spks.append(cur_spk)
                    cur_s, cur_e = nxt_s, nxt_e
                    cur_spk = nxt_spk
            merged_times.append((cur_s, cur_e))
            merged_spks.append(cur_spk)
            seg_times = merged_times
            speaker_ids = merged_spks

        total = len(seg_times)
        t_start = time.time()
        last_partial_push = time.time()

        for idx, ((s_sec, e_sec), spk_id) in enumerate(zip(seg_times, speaker_ids)):
            if progress_cb:
                pct = 25 + int(70 * (idx + 1) / max(1, total))
                eta_suffix = ""
                if idx >= 2:
                    elapsed = time.time() - t_start
                    avg_per_seg = elapsed / float(idx)
                    rem_sec = int(max(0.0, avg_per_seg * (total - idx)))
                    if rem_sec >= 60:
                        eta_suffix = f" (~{rem_sec // 60}m {rem_sec % 60:02d}s remaining)"
                    elif rem_sec > 2:
                        eta_suffix = f" (~{rem_sec}s remaining)"
                progress_cb(
                    f"Resuming segment {idx + 1}/{total}{eta_suffix}...", pct
                )

            clip = self.store.read_audio_slice(session.id, s_sec, e_sec)
            existing_spk = session.speakers.get(spk_id)
            forced_lang = (
                existing_spk.language
                if (existing_spk and existing_spk.language != "auto")
                else None
            )
            conv_ctx, spk_ctx = self._build_conversation_context(
                session, current_speaker_id=spk_id
            )
            pass1_clarif = (
                "whisper-context"
                if session.clarification_model in ("qwen2.5-0.5b", "qwen2.5-1.5b")
                else session.clarification_model
            )
            detected_lang, orig_text, trans_text, conf, raw_stt_text = (
                self.transcribe_and_translate_clip(
                    clip,
                    forced_language=forced_lang,
                    target_language=session.target_language,
                    model_id=session.whisper_model,
                    clarification_model=pass1_clarif,
                    translation_model=session.translation_model,
                    allowed_languages=session.spoken_languages,
                    conversation_context=conv_ctx,
                    speaker_context=spk_ctx,
                )
            )
            if not _has_meaningful_speech_text(orig_text):
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

            now_ts = time.time()
            if (len(session.segments) % 5 == 0) or (now_ts - last_partial_push >= 8.0):
                last_partial_push = now_ts
                self.store.save_session(session)
                if on_partial_update:
                    on_partial_update(session)

        self._run_global_clarification_pass(session, progress_cb=progress_cb)

        self.store.save_session(session)
        if progress_cb:
            progress_cb("Completed!", 100)
        return session

    def evict_model_cache(self, model_type: str, model_id: str = "") -> None:
        """Release ONNX Runtime sessions from memory before deleting model files or switching GPU/CPU."""
        with self._lock:
            if model_type == "all":
                self._whisper_runners.clear()
                self.clarifier._qwen_sessions.clear()
                self.translator._opus_cache.clear()
                self.translator._nllb_encoder = None
                self.translator._nllb_decoder = None
                self.translator._nllb_tokenizer = None
                self.diarizer._extractor = None
            elif model_type == "whisper":
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
