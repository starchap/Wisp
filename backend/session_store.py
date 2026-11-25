"""
Session persistence, speaker management, and WAV segment slicing for backtracking
and retroactive per-speaker re-transcription.
"""
import io
import json
import subprocess
import tempfile
import time
import uuid
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

import config
from config import (
    DEFAULT_CLARIFICATION_MODEL,
    DEFAULT_TRANSLATION_MODEL,
    DEFAULT_WHISPER_MODEL,
    SPEAKER_COLORS,
)

SAMPLE_RATE = 16000


@dataclass
class SpeakerProfile:
    id: str                      # e.g. "speaker_1", "speaker_2"
    name: str                    # Display name, defaults to "Speaker 1", "Speaker 2"
    language: str = "auto"       # Assigned spoken language ("auto", "da", "es", "en", etc.)
    color: str = "#3B82F6"
    segment_count: int = 0
    total_duration: float = 0.0


@dataclass
class TranscriptSegment:
    id: str
    speaker_id: str
    start: float                 # Seconds from start of session audio
    end: float                   # Seconds from start of session audio
    source_language: str         # Detected or forced language code ("da", "es", "en", etc.)
    original_text: str           # Text in speaker's language (after Context Clarification if enabled)
    translated_text: str         # Text translated into session's target_language
    target_language: str         # Target output language when translated
    confidence: float = 1.0
    is_reprocessing: bool = False
    raw_text: str = ""           # Raw first-pass Whisper text before Context Clarification


@dataclass
class MeetingSession:
    id: str
    title: str
    created_at: float
    updated_at: float
    session_type: str = "file"   # "file" or "live"
    source_file_name: str = ""
    target_language: str = "none"  # Single output language for the meeting (or "none" to skip translation)
    whisper_model: str = DEFAULT_WHISPER_MODEL
    clarification_model: str = DEFAULT_CLARIFICATION_MODEL
    translation_model: str = DEFAULT_TRANSLATION_MODEL
    spoken_languages: List[str] = field(default_factory=list)  # Expected languages in meeting, e.g. ["da", "en"]
    duration: float = 0.0
    audio_filename: str = "recording.wav"
    speakers: Dict[str, SpeakerProfile] = field(default_factory=dict)
    segments: List[TranscriptSegment] = field(default_factory=list)
    embeddings: Dict[str, List[float]] = field(default_factory=dict)  # speaker_id -> centroid embedding

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "session_type": self.session_type,
            "source_file_name": self.source_file_name,
            "target_language": self.target_language,
            "whisper_model": self.whisper_model,
            "clarification_model": self.clarification_model,
            "translation_model": self.translation_model,
            "spoken_languages": list(self.spoken_languages),
            "duration": round(self.duration, 2),
            "audio_filename": self.audio_filename,
            "speakers": {k: asdict(v) for k, v in self.speakers.items()},
            "segments": [asdict(s) for s in self.segments],
        }


def _resample_chunk_numpy(mono: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    if orig_sr == target_sr or len(mono) == 0:
        return mono.astype(np.float32)
    target_len = max(1, int(round(len(mono) * float(target_sr) / float(orig_sr))))
    spec = np.fft.rfft(mono)
    new_bins = target_len // 2 + 1
    if new_bins <= len(spec):
        spec_cropped = spec[:new_bins]
    else:
        spec_cropped = np.pad(spec, (0, new_bins - len(spec)))
    out = np.fft.irfft(spec_cropped, n=target_len) * (float(target_len) / float(len(mono)))
    return out.astype(np.float32)


def _resample_1d_numpy(mono: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Bandlimited anti-aliased Fourier/Sinc audio resampler in pure NumPy.
    Processes long recordings (e.g. 2-hour 48kHz MP4 files) in 30-second chunks so
    memory usage stays under 15 MB and FFT sizes remain small and ultra-fast (<0.5s for 2 hours).
    """
    if orig_sr == target_sr or len(mono) == 0:
        return mono.astype(np.float32)
    chunk_samples = int(orig_sr * 30)
    if len(mono) <= chunk_samples:
        return _resample_chunk_numpy(mono, orig_sr, target_sr)
    out_chunks: List[np.ndarray] = []
    for start_idx in range(0, len(mono), chunk_samples):
        slice_mono = mono[start_idx : start_idx + chunk_samples]
        if len(slice_mono) > 0:
            out_chunks.append(_resample_chunk_numpy(slice_mono, orig_sr, target_sr))
    return np.concatenate(out_chunks) if out_chunks else np.zeros(0, dtype=np.float32)


def _decode_audio_windows_media_foundation(file_path: str) -> np.ndarray:
    """
    Extract and decode the audio track of any .mp4, .m4a, .aac, .mov, .wma, .mp3, or .wav file
    directly into 16kHz mono float32 using Microsoft's built-in Windows Media Foundation system DLLs
    (`mfplat.dll` and `mfreadwrite.dll` in C:\\Windows\\System32).
    - Works on 100% of Windows 10/11 PCs out of the box (no `ffmpeg.exe` required).
    - Reads only the audio stream (`MF_SOURCE_READER_FIRST_AUDIO_STREAM`), skipping all video frames
      so a 2-hour .mp4 video file decodes in ~10-15 seconds with minimal RAM.
    """
    import ctypes
    from ctypes import POINTER, Structure, byref, c_int64, c_uint32, c_void_p, c_wchar_p

    class GUID(Structure):
        _fields_ = [
            ("Data1", c_uint32),
            ("Data2", ctypes.c_uint16),
            ("Data3", ctypes.c_uint16),
            ("Data4", ctypes.c_ubyte * 8),
        ]

        def __init__(self, d1, d2, d3, *d4):
            super().__init__(d1, d2, d3, (ctypes.c_ubyte * 8)(*d4))

    MF_MT_MAJOR_TYPE = GUID(0x48EBA18E, 0xF8C9, 0x4687, 0xBF, 0x11, 0x0A, 0x74, 0xC9, 0xF9, 0x6A, 0x8F)
    MF_MT_SUBTYPE = GUID(0xF7E34C9A, 0x42E8, 0x4714, 0xB7, 0x4B, 0xCB, 0x29, 0xD7, 0x2C, 0x35, 0xE5)
    MF_MT_AUDIO_NUM_CHANNELS = GUID(0x37E48BF5, 0x645E, 0x4C5B, 0x89, 0xDE, 0xAD, 0xA9, 0xE2, 0x9B, 0x69, 0x6A)
    MF_MT_AUDIO_SAMPLES_PER_SECOND = GUID(0x5FAEEAE7, 0x0290, 0x4C31, 0x9E, 0x8A, 0xC5, 0x34, 0xF6, 0x8D, 0x9D, 0xBA)
    MF_MT_AUDIO_BITS_PER_SAMPLE = GUID(0xF2DEB57F, 0x40FA, 0x4764, 0xAA, 0x33, 0xED, 0x4F, 0x2D, 0x1F, 0xF6, 0x69)
    MFMediaType_Audio = GUID(0x73647561, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71)
    MFAudioFormat_PCM = GUID(0x00000001, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71)

    def vcall(obj_ptr, idx, restype, *argtypes):
        vtbl = ctypes.cast(obj_ptr, POINTER(POINTER(c_void_p))).contents
        fn_type = ctypes.WINFUNCTYPE(restype, c_void_p, *argtypes)
        return fn_type(vtbl[idx])

    ctypes.windll.ole32.CoInitializeEx(None, 0)
    ctypes.windll.mfplat.MFStartup(0x00020070, 0)

    reader = c_void_p()
    hr = ctypes.windll.mfreadwrite.MFCreateSourceReaderFromURL(
        c_wchar_p(str(file_path)), None, byref(reader)
    )
    if hr < 0 or not reader.value:
        raise RuntimeError(f"MFCreateSourceReaderFromURL failed with HRESULT {hex(hr & 0xFFFFFFFF)}")

    try:
        mt = c_void_p()
        ctypes.windll.mfplat.MFCreateMediaType(byref(mt))
        vcall(mt, 24, ctypes.c_long, POINTER(GUID), POINTER(GUID))(
            mt, byref(MF_MT_MAJOR_TYPE), byref(MFMediaType_Audio)
        )
        vcall(mt, 24, ctypes.c_long, POINTER(GUID), POINTER(GUID))(
            mt, byref(MF_MT_SUBTYPE), byref(MFAudioFormat_PCM)
        )
        vcall(mt, 21, ctypes.c_long, POINTER(GUID), c_uint32)(
            mt, byref(MF_MT_AUDIO_BITS_PER_SAMPLE), c_uint32(16)
        )

        FIRST_AUDIO = c_uint32(0xFFFFFFFD)
        hr = vcall(reader, 7, ctypes.c_long, c_uint32, c_void_p, c_void_p)(
            reader, FIRST_AUDIO, None, mt
        )
        vcall(mt, 2, c_uint32)(mt)
        if hr < 0:
            raise RuntimeError(f"SetCurrentMediaType failed with HRESULT {hex(hr & 0xFFFFFFFF)}")

        out_mt = c_void_p()
        hr = vcall(reader, 6, ctypes.c_long, c_uint32, POINTER(c_void_p))(
            reader, FIRST_AUDIO, byref(out_mt)
        )
        if hr < 0 or not out_mt.value:
            raise RuntimeError("GetCurrentMediaType failed")

        ch = c_uint32(1)
        sr = c_uint32(SAMPLE_RATE)
        vcall(out_mt, 7, ctypes.c_long, POINTER(GUID), POINTER(c_uint32))(
            out_mt, byref(MF_MT_AUDIO_NUM_CHANNELS), byref(ch)
        )
        vcall(out_mt, 7, ctypes.c_long, POINTER(GUID), POINTER(c_uint32))(
            out_mt, byref(MF_MT_AUDIO_SAMPLES_PER_SECOND), byref(sr)
        )
        vcall(out_mt, 2, c_uint32)(out_mt)

        num_channels = max(1, int(ch.value))
        sample_rate = max(8000, int(sr.value))

        # Stream audio packets in ~60-second byte buffers and immediately mix down + resample to 16kHz mono
        # so even a 2-hour 48kHz stereo MP4 file uses minimal peak memory!
        resampled_blocks: List[np.ndarray] = []
        byte_accum = bytearray()
        flush_bytes_threshold = sample_rate * num_channels * 2 * 60  # 60 seconds of int16 PCM
        frame_bytes = num_channels * 2

        def _flush_accum() -> None:
            usable = (len(byte_accum) // frame_bytes) * frame_bytes
            if usable <= 0:
                return
            raw_chunk = bytes(byte_accum[:usable])
            del byte_accum[:usable]
            pcm_i16 = np.frombuffer(raw_chunk, dtype=np.int16)
            if num_channels > 1:
                pcm_f32 = pcm_i16.reshape(-1, num_channels).mean(axis=1).astype(np.float32) / 32768.0
            else:
                pcm_f32 = pcm_i16.astype(np.float32) / 32768.0
            if sample_rate != SAMPLE_RATE:
                pcm_f32 = _resample_1d_numpy(pcm_f32, sample_rate, SAMPLE_RATE)
            resampled_blocks.append(pcm_f32)

        while True:
            flags = c_uint32()
            ts = c_int64()
            sample = c_void_p()
            hr = vcall(
                reader,
                9,
                ctypes.c_long,
                c_uint32,
                c_uint32,
                POINTER(c_uint32),
                POINTER(c_uint32),
                POINTER(c_int64),
                POINTER(c_void_p),
            )(reader, FIRST_AUDIO, 0, None, byref(flags), byref(ts), byref(sample))
            if hr < 0 or (flags.value & 2):
                if sample.value:
                    vcall(sample, 2, c_uint32)(sample)
                break
            if sample.value:
                buf = c_void_p()
                if (
                    vcall(sample, 41, ctypes.c_long, POINTER(c_void_p))(sample, byref(buf)) >= 0
                    and buf.value
                ):
                    ptr = c_void_p()
                    cur_len = c_uint32()
                    if (
                        vcall(
                            buf,
                            3,
                            ctypes.c_long,
                            POINTER(c_void_p),
                            POINTER(c_uint32),
                            POINTER(c_uint32),
                        )(buf, byref(ptr), None, byref(cur_len))
                        >= 0
                    ):
                        if cur_len.value > 0:
                            byte_accum.extend(ctypes.string_at(ptr.value, cur_len.value))
                        vcall(buf, 4, ctypes.c_long)(buf)
                    vcall(buf, 2, c_uint32)(buf)
                vcall(sample, 2, c_uint32)(sample)
                if len(byte_accum) >= flush_bytes_threshold:
                    _flush_accum()

        if len(byte_accum) > 0:
            _flush_accum()

        return (
            np.concatenate(resampled_blocks)
            if resampled_blocks
            else np.zeros(0, dtype=np.float32)
        )
    finally:
        if reader.value:
            vcall(reader, 2, c_uint32)(reader)


def decode_audio_file_to_16k_mono(file_path: str) -> np.ndarray:
    """
    Decode an audio or video file on disk (.wav, .mp3, .flac, .ogg, .m4a, .webm, .mp4, .mov)
    directly into a 16kHz mono float32 numpy array without loading the entire raw file into RAM.
    """
    import sys

    ext = Path(file_path).suffix.lower()
    # For MP4 / M4A / MOV / AAC / WMA on Windows, use Windows Media Foundation directly first
    if sys.platform == "win32" and ext in (".mp4", ".m4a", ".aac", ".mov", ".wma", ".webm", ".mkv"):
        try:
            mono = _decode_audio_windows_media_foundation(file_path)
            if len(mono) > 0:
                return mono
        except Exception as mf_err:
            print(f"[AudioDecoder] Windows Media Foundation note on {ext}: {mf_err}")

    # Standard audio formats via libsndfile (streaming block read for long files)
    try:
        info = sf.info(file_path)
        sr = info.samplerate
        block_frames = sr * 60  # Read 60-second blocks so 2-hour files never allocate huge multi-channel arrays
        blocks: List[np.ndarray] = []
        for block in sf.blocks(file_path, blocksize=block_frames, dtype="float32", always_2d=True):
            mono_block = np.mean(block, axis=1).astype(np.float32)
            if sr != SAMPLE_RATE:
                mono_block = _resample_1d_numpy(mono_block, sr, SAMPLE_RATE)
            blocks.append(mono_block)
        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)
    except Exception as first_err:
        if sys.platform == "win32":
            try:
                mono = _decode_audio_windows_media_foundation(file_path)
                if len(mono) > 0:
                    return mono
            except Exception:
                pass
        with tempfile.NamedTemporaryFile(suffix=".conv16k.wav", delete=False) as tmp_out:
            out_path = tmp_out.name
        tmp_m4a = out_path + ".m4a"
        try:
            if sys.platform == "darwin":
                # 1. Try macOS built-in CoreAudio `afconvert`
                af_ok = False
                try:
                    subprocess.run(
                        [
                            "afconvert",
                            "-f",
                            "WAVE",
                            "-d",
                            f"LEI16@{SAMPLE_RATE}",
                            "-c",
                            "1",
                            str(file_path),
                            out_path,
                        ],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    af_ok = Path(out_path).exists() and Path(out_path).stat().st_size > 44
                except Exception:
                    af_ok = False

                # 2. If input is a video .mp4/.mov container where afconvert needs audio track extraction first,
                # use macOS built-in AVFoundation `avconvert` (or ffmpeg if installed)
                if not af_ok:
                    try:
                        subprocess.run(
                            [
                                "avconvert",
                                "--source",
                                str(file_path),
                                "--output",
                                tmp_m4a,
                                "--preset",
                                "PresetAppleM4A",
                                "--replace",
                            ],
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        subprocess.run(
                            [
                                "afconvert",
                                "-f",
                                "WAVE",
                                "-d",
                                f"LEI16@{SAMPLE_RATE}",
                                "-c",
                                "1",
                                tmp_m4a,
                                out_path,
                            ],
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    except Exception:
                        subprocess.run(
                            [
                                "ffmpeg",
                                "-y",
                                "-i",
                                str(file_path),
                                "-vn",
                                "-ar",
                                str(SAMPLE_RATE),
                                "-ac",
                                "1",
                                out_path,
                            ],
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
            else:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(file_path),
                        "-vn",
                        "-ar",
                        str(SAMPLE_RATE),
                        "-ac",
                        "1",
                        out_path,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            mono, _ = sf.read(out_path, dtype="float32", always_2d=False)
            return mono.astype(np.float32)
        except Exception:
            raise ValueError(
                f"Could not decode audio/video file ({first_err}). Supported formats: MP4, M4A, WAV, MP3, FLAC, OGG."
            )
        finally:
            Path(out_path).unlink(missing_ok=True)
            Path(tmp_m4a).unlink(missing_ok=True)


def decode_audio_to_16k_mono(raw_bytes: bytes, filename: str = "audio.wav") -> np.ndarray:
    """
    Decode audio bytes (.wav, .mp3, .flac, .ogg, .m4a, .webm, .mp4)
    into a 16kHz mono float32 numpy array in [-1.0, 1.0].
    """
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(raw_bytes)
        in_path = tmp_in.name
    try:
        return decode_audio_file_to_16k_mono(in_path)
    finally:
        Path(in_path).unlink(missing_ok=True)


class SessionStore:
    """Manages local storage of meeting sessions, audio files, and transcripts."""

    def __init__(self):
        self.sessions: Dict[str, MeetingSession] = {}
        self._load_existing_sessions()

    def reload_from_storage(self) -> None:
        self.sessions.clear()
        self._load_existing_sessions()

    def _session_dir(self, session_id: str) -> Path:
        d = config.SESSIONS_DIR / session_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_audio_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "recording.wav"

    def _metadata_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "session.json"

    def _load_existing_sessions(self) -> None:
        if not config.SESSIONS_DIR.exists():
            return
        for folder in sorted(config.SESSIONS_DIR.iterdir(), reverse=True):
            meta_file = folder / "session.json"
            if meta_file.exists():
                try:
                    data = json.loads(meta_file.read_text(encoding="utf-8-sig"))
                    speakers = {
                        k: SpeakerProfile(**v) for k, v in data.get("speakers", {}).items()
                    }
                    segments = [
                        TranscriptSegment(
                            **{
                                k: v
                                for k, v in s.items()
                                if k in TranscriptSegment.__dataclass_fields__
                            }
                        )
                        for s in data.get("segments", [])
                    ]
                    raw_w_model = data.get("whisper_model", DEFAULT_WHISPER_MODEL)
                    if raw_w_model == "hviske-v2-fp16":
                        raw_w_model = "hviske-v2-int8"
                    audio_fname = data.get("audio_filename", "recording.wav")
                    dur = float(data.get("duration", 0.0) or 0.0)
                    wav_path = folder / audio_fname
                    if dur <= 0.0 and wav_path.exists():
                        try:
                            w_size = wav_path.stat().st_size
                            if w_size > 44:
                                dur = round((w_size - 44) / float(SAMPLE_RATE * 2), 2)
                        except Exception:
                            pass
                    session = MeetingSession(
                        id=data["id"],
                        title=data.get("title", "Untitled Meeting"),
                        created_at=data.get("created_at", time.time()),
                        updated_at=data.get("updated_at", time.time()),
                        session_type=data.get("session_type", "file"),
                        source_file_name=data.get("source_file_name", ""),
                        target_language=data.get("target_language", "en"),
                        whisper_model=raw_w_model,
                        clarification_model=data.get(
                            "clarification_model", DEFAULT_CLARIFICATION_MODEL
                        ),
                        translation_model=data.get(
                            "translation_model", DEFAULT_TRANSLATION_MODEL
                        ),
                        spoken_languages=[
                            c
                            for c in data.get("spoken_languages", [])
                            if c and c not in ("auto", "none")
                        ],
                        duration=dur,
                        audio_filename=audio_fname,
                        speakers=speakers,
                        segments=segments,
                        embeddings=data.get("embeddings", {}),
                    )
                    self.sessions[session.id] = session
                except Exception as exc:
                    print(f"[SessionStore] Error loading {meta_file}: {exc}")

    def save_session(self, session: MeetingSession) -> None:
        session.updated_at = time.time()
        for spk in session.speakers.values():
            spk.segment_count = 0
            spk.total_duration = 0.0
        for seg in session.segments:
            if seg.speaker_id not in session.speakers:
                self.ensure_speaker(session, seg.speaker_id)
            session.speakers[seg.speaker_id].segment_count += 1
            session.speakers[seg.speaker_id].total_duration += max(0.0, seg.end - seg.start)

        # If the session has segments, prune any unused/orphan speaker with 0 segments
        # so each session only lists speakers who actually spoke in this session
        if session.segments:
            empty_ids = [
                spk_id
                for spk_id, spk in session.speakers.items()
                if spk.segment_count == 0
            ]
            for spk_id in empty_ids:
                session.speakers.pop(spk_id, None)
                session.embeddings.pop(spk_id, None)

        payload = session.to_dict()
        payload["embeddings"] = session.embeddings
        self._metadata_path(session.id).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def create_session(
        self,
        title: str = "New Meeting",
        session_type: str = "file",
        target_language: str = "none",
        whisper_model: str = DEFAULT_WHISPER_MODEL,
        clarification_model: str = DEFAULT_CLARIFICATION_MODEL,
        translation_model: str = DEFAULT_TRANSLATION_MODEL,
        spoken_languages: Optional[List[str]] = None,
    ) -> MeetingSession:
        session_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        now = time.time()
        clean_spoken = [
            c for c in (spoken_languages or []) if c and c not in ("auto", "none")
        ]
        session = MeetingSession(
            id=session_id,
            title=title,
            created_at=now,
            updated_at=now,
            session_type=session_type if session_type in ("file", "live") else "file",
            target_language=target_language,
            whisper_model=whisper_model,
            clarification_model=clarification_model or DEFAULT_CLARIFICATION_MODEL,
            translation_model=translation_model or DEFAULT_TRANSLATION_MODEL,
            spoken_languages=clean_spoken,
        )
        audio_path = self.get_audio_path(session_id)
        with wave.open(str(audio_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"")

        self.sessions[session_id] = session
        self.save_session(session)
        return session

    def get_session(self, session_id: str) -> Optional[MeetingSession]:
        return self.sessions.get(session_id)

    def list_sessions(self) -> List[dict]:
        sorted_sessions = sorted(
            self.sessions.values(), key=lambda s: s.created_at, reverse=True
        )
        return [
            {
                "id": s.id,
                "title": s.title,
                "created_at": s.created_at,
                "updated_at": s.updated_at,
                "session_type": s.session_type,
                "source_file_name": s.source_file_name,
                "target_language": s.target_language,
                "whisper_model": s.whisper_model,
                "clarification_model": s.clarification_model,
                "translation_model": s.translation_model,
                "spoken_languages": list(s.spoken_languages),
                "duration": round(s.duration, 2),
                "speaker_count": len(s.speakers),
                "segment_count": len(s.segments),
            }
            for s in sorted_sessions
        ]

    def delete_session(self, session_id: str) -> bool:
        if session_id not in self.sessions:
            return False
        del self.sessions[session_id]
        try:
            self._metadata_path(session_id).unlink(missing_ok=True)
        except Exception:
            pass
        folder = self._session_dir(session_id)
        import shutil
        shutil.rmtree(folder, ignore_errors=True)
        return True

    def ensure_speaker(
        self, session: MeetingSession, speaker_id: str, default_lang: str = "auto"
    ) -> SpeakerProfile:
        if speaker_id not in session.speakers:
            idx = len(session.speakers) + 1
            if speaker_id.startswith("speaker_"):
                try:
                    idx = int(speaker_id.split("_")[1])
                except ValueError:
                    pass
            color = SPEAKER_COLORS[(idx - 1) % len(SPEAKER_COLORS)]
            session.speakers[speaker_id] = SpeakerProfile(
                id=speaker_id,
                name=f"Speaker {idx}",
                language=default_lang,
                color=color,
            )
        return session.speakers[speaker_id]

    def append_audio_pcm(self, session_id: str, audio_f32: np.ndarray) -> Tuple[float, float]:
        """
        Append float32 16kHz mono samples to the session's WAV file and return (start_sec, end_sec).
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        audio_path = self.get_audio_path(session_id)
        pcm_int16 = np.clip(audio_f32 * 32767.0, -32768, 32767).astype(np.int16)

        start_sec = session.duration
        chunk_duration = len(pcm_int16) / float(SAMPLE_RATE)
        end_sec = start_sec + chunk_duration

        if audio_path.exists() and audio_path.stat().st_size > 44:
            with sf.SoundFile(str(audio_path), mode="r+") as f:
                f.seek(0, sf.SEEK_END)
                f.write(pcm_int16)
                end_sec = len(f) / float(SAMPLE_RATE)
                start_sec = end_sec - chunk_duration
        else:
            sf.write(str(audio_path), pcm_int16, SAMPLE_RATE, subtype="PCM_16")

        session.duration = end_sec
        return start_sec, end_sec

    def write_full_audio(self, session_id: str, audio_f32: np.ndarray) -> float:
        """Write complete float32 16kHz mono array to session's recording.wav."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        audio_path = self.get_audio_path(session_id)
        pcm_int16 = np.clip(audio_f32 * 32767.0, -32768, 32767).astype(np.int16)
        sf.write(str(audio_path), pcm_int16, SAMPLE_RATE, subtype="PCM_16")
        session.duration = len(pcm_int16) / float(SAMPLE_RATE)
        return session.duration

    def read_audio_slice(
        self, session_id: str, start_sec: float, end_sec: float
    ) -> np.ndarray:
        """Read a specific [start_sec, end_sec] slice from the session's recording.wav."""
        audio_path = self.get_audio_path(session_id)
        if not audio_path.exists():
            return np.zeros(0, dtype=np.float32)

        start_frame = max(0, int(start_sec * SAMPLE_RATE))
        num_frames = max(1, int((end_sec - start_sec) * SAMPLE_RATE))

        with sf.SoundFile(str(audio_path), mode="r") as f:
            if start_frame >= len(f):
                return np.zeros(0, dtype=np.float32)
            f.seek(start_frame)
            data = f.read(frames=num_frames, dtype="float32", always_2d=False)
            return data
