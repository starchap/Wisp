"""
Local Speech-to-Text (Sherpa-ONNX Whisper INT8) and Specialized Offline Neural Machine Translation
(Helsinki-NLP OPUS-MT MarianMT ONNX INT8 focused on Danish <-> English + Spanish/German,
Meta NLLB-200 ONNX INT8, and Whisper Native English Translation) Pipeline for Wishper.
"""
import json
import threading
import uuid
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import sentencepiece as spm
import sherpa_onnx
from tokenizers import Tokenizer

import config as app_config
from config import (
    DEFAULT_TRANSLATION_MODEL,
    DEFAULT_WHISPER_MODEL,
    LANGUAGE_NLLB_MAP,
    OPUS_PAIR_REPOS,
)
from diarization import LocalSpeakerDiarizer
from models_manager import (
    get_opus_pair_dir,
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
        if not text or source_lang == target_lang or not target_lang:
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


class LocalSpeechPipeline:
    """
    Coordinates Sherpa-ONNX Whisper (speech recognition + spoken language ID),
    LocalSpeakerDiarizer (speaker recognition), and LocalONNXTranslationEngine
    (OPUS-MT Danish <-> English Specialist, Meta NLLB-200, and Whisper Direct Translate).
    """

    def __init__(self, session_store: SessionStore):
        self.store = session_store
        self.diarizer = LocalSpeakerDiarizer()
        self.translator = LocalONNXTranslationEngine()
        self._recognizers: Dict[Tuple[str, str, str], sherpa_onnx.OfflineRecognizer] = {}
        self._lang_detectors: Dict[str, sherpa_onnx.SpokenLanguageIdentification] = {}
        self._lock = threading.Lock()
        self.device = "onnx-cpu"
        self.compute_type = "int8"
        print(
            "[SpeechPipeline] Initialized Sherpa-ONNX Whisper + 3D-Speaker + OPUS-MT (DA<->EN) & NLLB-200 Engine."
        )

    def _get_lang_detector(
        self, model_id: str = DEFAULT_WHISPER_MODEL
    ) -> Optional[sherpa_onnx.SpokenLanguageIdentification]:
        if model_id in self._lang_detectors:
            return self._lang_detectors[model_id]
        with self._lock:
            if model_id in self._lang_detectors:
                return self._lang_detectors[model_id]
            try:
                paths = resolve_local_whisper_model(model_id)
                slid_cfg = sherpa_onnx.SpokenLanguageIdentificationConfig(
                    whisper=sherpa_onnx.SpokenLanguageIdentificationWhisperConfig(
                        encoder=str(paths["encoder"]),
                        decoder=str(paths["decoder"]),
                        tail_paddings=500,
                    ),
                    num_threads=2,
                    debug=False,
                    provider="cpu",
                )
                detector = sherpa_onnx.SpokenLanguageIdentification(slid_cfg)
                self._lang_detectors[model_id] = detector
                return detector
            except Exception as exc:
                print(f"[SpeechPipeline] Language detector init warning: {exc}")
                return None

    def detect_spoken_language(
        self, audio_16k: np.ndarray, model_id: str = DEFAULT_WHISPER_MODEL
    ) -> str:
        detector = self._get_lang_detector(model_id)
        if detector is None:
            return "da"
        try:
            norm_audio = self._normalize_speech_clip(audio_16k)
            stream = detector.create_stream()
            stream.accept_waveform(SAMPLE_RATE, norm_audio.tolist())
            lang = detector.compute(stream)
            return lang.strip() if lang else "da"
        except Exception as exc:
            print(f"[SpeechPipeline] Language detect fallback to 'da': {exc}")
            return "da"

    @staticmethod
    def _normalize_speech_clip(audio_16k: np.ndarray) -> np.ndarray:
        """Peak-normalize quiet microphone/loopback clips so INT8 Whisper models hear speech clearly."""
        if len(audio_16k) == 0:
            return audio_16k
        centered = audio_16k - float(np.mean(audio_16k))
        peak = float(np.max(np.abs(centered)))
        if 0.002 < peak < 0.75:
            return np.clip(centered * (0.82 / peak), -1.0, 1.0).astype(np.float32)
        return centered.astype(np.float32)

    def _get_whisper_recognizer(
        self,
        model_id: str = DEFAULT_WHISPER_MODEL,
        language: str = "en",
        task: str = "transcribe",
    ) -> sherpa_onnx.OfflineRecognizer:
        lang_code = "" if (not language or language == "auto") else language
        key = (model_id, lang_code, task)
        if key in self._recognizers:
            return self._recognizers[key]

        with self._lock:
            if key in self._recognizers:
                return self._recognizers[key]

            paths = resolve_local_whisper_model(model_id)
            rec = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=str(paths["encoder"]),
                decoder=str(paths["decoder"]),
                tokens=str(paths["tokens"]),
                language=lang_code,
                task=task,
                num_threads=4,
                decoding_method="greedy_search",
                debug=False,
                provider="cpu",
            )
            self._recognizers[key] = rec
            return rec

    def _run_whisper_clip(
        self,
        audio_16k: np.ndarray,
        model_id: str,
        language: str,
        task: str = "transcribe",
    ) -> str:
        norm_audio = self._normalize_speech_clip(audio_16k)
        rec = self._get_whisper_recognizer(
            model_id=model_id, language=language, task=task
        )
        stream = rec.create_stream()
        stream.accept_waveform(SAMPLE_RATE, norm_audio.tolist())
        rec.decode_stream(stream)
        return stream.result.text.strip()

    def transcribe_and_translate_clip(
        self,
        audio_16k: np.ndarray,
        forced_language: Optional[str],
        target_language: str,
        model_id: str = DEFAULT_WHISPER_MODEL,
        translation_model: str = DEFAULT_TRANSLATION_MODEL,
    ) -> Tuple[str, str, str, float]:
        """
        Transcribe a single audio clip in `forced_language` (or auto-detect if None/"auto"),
        and translate to `target_language` using `translation_model`.
        Returns: (detected_source_lang, original_text, translated_text, confidence)
        """
        if len(audio_16k) < int(SAMPLE_RATE * 0.25):
            return forced_language or "en", "", "", 0.0

        if not forced_language or forced_language == "auto":
            detected_lang = self.detect_spoken_language(audio_16k, model_id=model_id)
        else:
            detected_lang = forced_language

        # Step 1: Transcribe in the speaker's spoken language
        original_text = self._run_whisper_clip(
            audio_16k,
            model_id=model_id,
            language=detected_lang,
            task="transcribe",
        )
        if not original_text:
            return detected_lang, "", "", 1.0

        # Step 2: Translate into single output `target_language` using the selected local AI translation model
        if detected_lang == target_language:
            translated_text = original_text
        elif translation_model == "whisper-native" and target_language == "en":
            w_trans = self._run_whisper_clip(
                audio_16k,
                model_id=model_id,
                language=detected_lang,
                task="translate",
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

        return detected_lang, original_text, translated_text, 1.0

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
            if effective_forced_lang is None and len(sub_audio) < int(SAMPLE_RATE * 2.2):
                prev_spk_segs = [s for s in session.segments if s.speaker_id == spk_id and s.source_language]
                if prev_spk_segs:
                    effective_forced_lang = prev_spk_segs[-1].source_language

            detected_lang, orig_text, trans_text, confidence = (
                self.transcribe_and_translate_clip(
                    sub_audio,
                    forced_language=effective_forced_lang,
                    target_language=session.target_language,
                    model_id=session.whisper_model,
                    translation_model=session.translation_model,
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
            detected_lang, orig_text, trans_text, conf = (
                self.transcribe_and_translate_clip(
                    clip,
                    forced_language=forced_lang,
                    target_language=session.target_language,
                    model_id=session.whisper_model,
                    translation_model=session.translation_model,
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
            )
            session.segments.append(seg_obj)

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

        for seg in session.segments:
            if seg.speaker_id != speaker_id:
                continue

            clip = self.store.read_audio_slice(session.id, seg.start, seg.end)
            if len(clip) < int(SAMPLE_RATE * 0.25):
                continue

            detected_lang, orig_text, trans_text, conf = (
                self.transcribe_and_translate_clip(
                    clip,
                    forced_language=None if speaker_lang == "auto" else speaker_lang,
                    target_language=session.target_language,
                    model_id=session.whisper_model,
                    translation_model=session.translation_model,
                )
            )
            if orig_text:
                seg.source_language = detected_lang
                seg.original_text = orig_text
                seg.translated_text = trans_text
                seg.target_language = session.target_language
                seg.confidence = round(conf, 3)
            updated.append(seg)

        self.store.save_session(session)
        return updated

    def retranslate_all_segments(
        self, session: MeetingSession
    ) -> List[TranscriptSegment]:
        """
        When the user changes the single target output language or the translation model for the meeting,
        re-translate all existing segments into `session.target_language`.
        """
        for seg in session.segments:
            if seg.source_language == session.target_language:
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
        Re-transcribe and re-translate all existing segments of a session using
        its current `whisper_model`, `translation_model`, and `target_language`
        (preserving speaker profiles, names, and per-speaker spoken languages).
        """
        if not session.segments:
            audio_path = self.store.get_audio_path(session.id)
            if audio_path.exists():
                raw = audio_path.read_bytes()
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
                f"Loading Whisper ({session.whisper_model}) & Translation ({session.translation_model})...",
                10,
            )
        resolve_local_whisper_model(session.whisper_model)

        total = len(session.segments)
        for idx, seg in enumerate(session.segments):
            if progress_cb:
                pct = 15 + int(80 * (idx + 1) / max(1, total))
                progress_cb(
                    f"Re-transcribing & translating segment {idx + 1}/{total} ({session.whisper_model})...",
                    pct,
                )

            clip = self.store.read_audio_slice(session.id, seg.start, seg.end)
            if len(clip) < int(SAMPLE_RATE * 0.25):
                continue

            spk = session.speakers.get(seg.speaker_id)
            forced_lang = (
                spk.language if (spk and spk.language != "auto") else None
            )
            detected_lang, orig_text, trans_text, conf = (
                self.transcribe_and_translate_clip(
                    clip,
                    forced_language=forced_lang,
                    target_language=session.target_language,
                    model_id=session.whisper_model,
                    translation_model=session.translation_model,
                )
            )
            if orig_text:
                seg.source_language = detected_lang
                seg.original_text = orig_text
                seg.translated_text = trans_text
                seg.target_language = session.target_language
                seg.confidence = round(conf, 3)

        self.store.save_session(session)
        if progress_cb:
            progress_cb("Completed re-transcription & translation!", 100)
        return session

    def evict_model_cache(self, model_type: str, model_id: str = "") -> None:
        """Release ONNX Runtime sessions from memory before deleting model files on disk."""
        with self._lock:
            if model_type == "whisper":
                keys_to_remove = [
                    k for k in self._recognizers if k[0] == model_id
                ]
                for k in keys_to_remove:
                    self._recognizers.pop(k, None)
                self._lang_detectors.pop(model_id, None)
            elif model_type in ("opus", "opus-mt-da-en", "translation"):
                self.translator._opus_cache.clear()
            elif model_type in ("nllb", "nllb-200-600M"):
                self.translator._nllb_encoder = None
                self.translator._nllb_decoder = None
                self.translator._nllb_tokenizer = None
            elif model_type == "speaker":
                self.diarizer._extractor = None
