"""
Cross-Platform Native Audio Capture Manager (Windows WASAPI Loopback, Linux PulseAudio/PipeWire, macOS CoreAudio)
using the `soundcard` library, plus support for streaming audio chunks from the Electron/Browser frontend.
Allows capturing:
1. System Loopback Audio (e.g., Teams / Slack Huddle / Zoom output)
2. Microphone Input
3. Dual Mix (Loopback + Microphone simultaneously!)
"""
import threading
import time
from typing import Callable, Dict, List, Optional
import numpy as np

try:
    import soundcard as sc
    HAS_SOUNDCARD = True
except Exception as exc:
    print(f"[AudioCapture] soundcard library warning: {exc}")
    HAS_SOUNDCARD = False

from session_store import SAMPLE_RATE


def list_audio_devices() -> Dict[str, List[Dict]]:
    """List available microphones and system loopback speakers across Windows, Linux, and macOS."""
    mics: List[Dict] = []
    loopbacks: List[Dict] = []

    if not HAS_SOUNDCARD:
        return {"microphones": mics, "loopbacks": loopbacks}

    try:
        default_mic = sc.default_microphone()
        default_mic_id = default_mic.id if default_mic else None
    except Exception:
        default_mic_id = None

    try:
        default_spk = sc.default_speaker()
        default_spk_id = default_spk.id if default_spk else None
    except Exception:
        default_spk_id = None

    try:
        for m in sc.all_microphones(include_loopback=False):
            mics.append(
                {
                    "id": str(m.id),
                    "name": str(m.name),
                    "is_default": str(m.id) == str(default_mic_id),
                    "type": "microphone",
                }
            )
    except Exception as exc:
        print(f"[AudioCapture] Error listing microphones: {exc}")

    try:
        # include_loopback=True exposes system output speakers as loopback capture sources (WASAPI/Pulse)
        for lb in sc.all_microphones(include_loopback=True):
            if getattr(lb, "isloopback", False) or str(lb.id) not in {m["id"] for m in mics}:
                loopbacks.append(
                    {
                        "id": str(lb.id),
                        "name": f"{lb.name} (System Audio Loopback)",
                        "is_default": str(lb.id) == str(default_spk_id),
                        "type": "loopback",
                    }
                )
    except Exception as exc:
        print(f"[AudioCapture] Error listing loopback devices: {exc}")

    return {"microphones": mics, "loopbacks": loopbacks}


class BackendLiveRecorder:
    """
    Captures live audio from a system loopback device (Teams/Huddle) and/or microphone
    in the background and emits 16kHz mono float32 chunks every `chunk_seconds` for real-time diarization + ASR.
    """

    def __init__(
        self,
        chunk_callback: Callable[[np.ndarray, float], None],
        chunk_seconds: float = 4.0,
    ):
        self.chunk_callback = chunk_callback
        self.chunk_seconds = chunk_seconds
        self.is_recording = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(
        self,
        loopback_id: Optional[str] = None,
        mic_id: Optional[str] = None,
    ) -> None:
        if self.is_recording:
            return
        self._stop_event.clear()
        self.is_recording = True
        self._thread = threading.Thread(
            target=self._record_loop,
            args=(loopback_id, mic_id),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self.is_recording = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None

    def _find_device(self, dev_id: Optional[str], include_loopback: bool):
        if not HAS_SOUNDCARD or not dev_id:
            return None
        try:
            for d in sc.all_microphones(include_loopback=include_loopback):
                if str(d.id) == str(dev_id):
                    return d
        except Exception:
            pass
        return None

    def _record_loop(self, loopback_id: Optional[str], mic_id: Optional[str]) -> None:
        if not HAS_SOUNDCARD:
            self.is_recording = False
            return

        lb_dev = self._find_device(loopback_id, include_loopback=True) if loopback_id else None
        mic_dev = self._find_device(mic_id, include_loopback=False) if mic_id else None

        if not lb_dev and not mic_dev:
            # Default to system loopback if available, else default mic
            try:
                all_lbs = [
                    d
                    for d in sc.all_microphones(include_loopback=True)
                    if getattr(d, "isloopback", False)
                ]
                if all_lbs:
                    lb_dev = all_lbs[0]
                else:
                    mic_dev = sc.default_microphone()
            except Exception:
                pass

        frames_per_read = int(SAMPLE_RATE * 0.25)  # 250ms blocks
        target_samples = int(SAMPLE_RATE * self.chunk_seconds)
        buffer: List[np.ndarray] = []
        current_samples = 0

        try:
            if lb_dev and mic_dev:
                with lb_dev.recorder(samplerate=SAMPLE_RATE, channels=1) as rec_lb, mic_dev.recorder(
                    samplerate=SAMPLE_RATE, channels=1
                ) as rec_mic:
                    while not self._stop_event.is_set():
                        data_lb = rec_lb.record(numframes=frames_per_read)
                        data_mic = rec_mic.record(numframes=frames_per_read)
                        min_len = min(len(data_lb), len(data_mic))
                        mixed = (
                            data_lb[:min_len].reshape(-1) + data_mic[:min_len].reshape(-1)
                        ) * 0.5
                        buffer.append(mixed.astype(np.float32))
                        current_samples += len(mixed)
                        if current_samples >= target_samples:
                            chunk = np.concatenate(buffer)
                            buffer = []
                            current_samples = 0
                            rms = float(np.sqrt(np.mean(chunk**2)))
                            self.chunk_callback(chunk, rms)
            else:
                active_dev = lb_dev or mic_dev
                if not active_dev:
                    self.is_recording = False
                    return
                with active_dev.recorder(samplerate=SAMPLE_RATE, channels=1) as rec:
                    while not self._stop_event.is_set():
                        data = rec.record(numframes=frames_per_read).reshape(-1).astype(np.float32)
                        buffer.append(data)
                        current_samples += len(data)
                        if current_samples >= target_samples:
                            chunk = np.concatenate(buffer)
                            buffer = []
                            current_samples = 0
                            rms = float(np.sqrt(np.mean(chunk**2)))
                            self.chunk_callback(chunk, rms)
        except Exception as exc:
            print(f"[BackendLiveRecorder] Recording loop error: {exc}")
        finally:
            if buffer:
                chunk = np.concatenate(buffer)
                if len(chunk) > int(SAMPLE_RATE * 0.3):
                    rms = float(np.sqrt(np.mean(chunk**2)))
                    self.chunk_callback(chunk, rms)
            self.is_recording = False
