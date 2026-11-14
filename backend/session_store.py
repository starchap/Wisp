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
from scipy.signal import resample_poly

import config
from config import (
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
    original_text: str           # Text in speaker's language
    translated_text: str         # Text translated into session's target_language
    target_language: str         # Target output language when translated
    confidence: float = 1.0
    is_reprocessing: bool = False


@dataclass
class MeetingSession:
    id: str
    title: str
    created_at: float
    updated_at: float
    session_type: str = "file"   # "file" or "live"
    source_file_name: str = ""
    target_language: str = "en"  # Single output language for the meeting
    whisper_model: str = DEFAULT_WHISPER_MODEL
    translation_model: str = DEFAULT_TRANSLATION_MODEL
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
            "translation_model": self.translation_model,
            "duration": round(self.duration, 2),
            "audio_filename": self.audio_filename,
            "speakers": {k: asdict(v) for k, v in self.speakers.items()},
            "segments": [asdict(s) for s in self.segments],
        }


def decode_audio_to_16k_mono(raw_bytes: bytes, filename: str = "audio.wav") -> np.ndarray:
    """
    Decode audio bytes (.wav, .mp3, .flac, .ogg, .m4a, .webm, .mp4)
    into a 16kHz mono float32 numpy array in [-1.0, 1.0].
    Uses `soundfile` (libsndfile supports WAV, MP3, FLAC, OGG natively) with `scipy.signal.resample_poly`.
    """
    try:
        data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=True)
        # Mix down to mono
        mono = np.mean(data, axis=1).astype(np.float32)
        if sr != SAMPLE_RATE:
            gcd = np.gcd(sr, SAMPLE_RATE)
            up = SAMPLE_RATE // gcd
            down = sr // gcd
            mono = resample_poly(mono, up, down).astype(np.float32)
        return mono
    except Exception as first_err:
        # Fallback to native OS audio converters (macOS `afconvert` or `ffmpeg`) for .m4a / .mp4 / .aac
        suffix = Path(filename).suffix or ".m4a"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
            tmp_in.write(raw_bytes)
            in_path = tmp_in.name
        out_path = in_path + ".wav"
        try:
            import sys
            if sys.platform == "darwin":
                # macOS built-in CoreAudio converter for .m4a / .mp4 / .aac
                subprocess.run(
                    [
                        "afconvert",
                        "-f",
                        "WAVE",
                        "-d",
                        f"LEI16@{SAMPLE_RATE}",
                        "-c",
                        "1",
                        in_path,
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
                        in_path,
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
            mono, _ = sf.read(out_path, dtype="float32", always_2d=False)
            return mono.astype(np.float32)
        except Exception:
            raise ValueError(
                f"Could not decode audio file ({first_err}). Supported formats out-of-the-box: WAV, MP3, M4A, FLAC, OGG."
            )
        finally:
            Path(in_path).unlink(missing_ok=True)
            Path(out_path).unlink(missing_ok=True)


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
                    data = json.loads(meta_file.read_text(encoding="utf-8"))
                    speakers = {
                        k: SpeakerProfile(**v) for k, v in data.get("speakers", {}).items()
                    }
                    segments = [
                        TranscriptSegment(**s) for s in data.get("segments", [])
                    ]
                    session = MeetingSession(
                        id=data["id"],
                        title=data.get("title", "Untitled Meeting"),
                        created_at=data.get("created_at", time.time()),
                        updated_at=data.get("updated_at", time.time()),
                        session_type=data.get("session_type", "file"),
                        source_file_name=data.get("source_file_name", ""),
                        target_language=data.get("target_language", "en"),
                        whisper_model=data.get("whisper_model", DEFAULT_WHISPER_MODEL),
                        translation_model=data.get("translation_model", DEFAULT_TRANSLATION_MODEL),
                        duration=data.get("duration", 0.0),
                        audio_filename=data.get("audio_filename", "recording.wav"),
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
        target_language: str = "en",
        whisper_model: str = DEFAULT_WHISPER_MODEL,
        translation_model: str = DEFAULT_TRANSLATION_MODEL,
    ) -> MeetingSession:
        session_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        now = time.time()
        session = MeetingSession(
            id=session_id,
            title=title,
            created_at=now,
            updated_at=now,
            session_type=session_type if session_type in ("file", "live") else "file",
            target_language=target_language,
            whisper_model=whisper_model,
            translation_model=translation_model,
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
                "translation_model": s.translation_model,
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
