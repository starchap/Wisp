"""
Cross-Platform Native Audio Capture Manager (Windows WASAPI Loopback, Linux PulseAudio/PipeWire, macOS CoreAudio)
using the `soundcard` library, plus support for streaming audio chunks from the Electron/Browser frontend.
Allows capturing:
1. System Loopback Audio (e.g., Teams / Slack Huddle / Zoom output)
2. Microphone Input
3. Dual Mix (Loopback + Microphone simultaneously without WASAPI loopback blocking the mic!)
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


def _to_mono_1d(raw: np.ndarray) -> np.ndarray:
    """
    Safely downmix multi-channel (N, C) audio from soundcard into a 1D mono float32 array of length N.
    Never use .reshape(-1) on a 2D stereo array, as that interleaves channels and halves the pitch/speed!
    """
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape[1] > 1:
            return np.mean(arr, axis=1, dtype=np.float32)
        return arr[:, 0]
    return arr.reshape(-1)


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
    in the background, emits real-time RMS audio levels every 250ms, and emits 16kHz mono
    float32 speech chunks for real-time diarization + ASR.
    """

    def __init__(
        self,
        chunk_callback: Callable[[np.ndarray, float], None],
        level_callback: Optional[Callable[[float], None]] = None,
        chunk_seconds: float = 4.5,
    ):
        self.chunk_callback = chunk_callback
        self.level_callback = level_callback
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
            try:
                mic_dev = sc.default_microphone()
            except Exception:
                pass
            if not mic_dev:
                try:
                    all_lbs = [
                        d
                        for d in sc.all_microphones(include_loopback=True)
                        if getattr(d, "isloopback", False)
                    ]
                    if all_lbs:
                        lb_dev = all_lbs[0]
                except Exception:
                    pass

        frames_per_read = int(SAMPLE_RATE * 0.25)  # 250ms blocks
        target_samples = int(SAMPLE_RATE * self.chunk_seconds)

        # Shared ring buffers protected by lock so WASAPI loopback (which blocks when speakers are silent)
        # NEVER blocks the microphone thread!
        lock = threading.Lock()
        latest_lb: List[np.ndarray] = []
        latest_mic: List[np.ndarray] = []

        def _reader_worker(dev, target_list: List[np.ndarray], label: str):
            try:
                # Initialize COM on Windows worker thread if needed
                try:
                    import pythoncom  # type: ignore
                    pythoncom.CoInitialize()
                except Exception:
                    pass
                with dev.recorder(samplerate=SAMPLE_RATE) as rec:
                    while not self._stop_event.is_set():
                        raw = rec.record(numframes=frames_per_read)
                        mono = _to_mono_1d(raw)
                        if len(mono) > 0:
                            with lock:
                                target_list.append(mono)
            except Exception as exc:
                print(f"[BackendLiveRecorder] {label} reader warning: {exc}")

        threads: List[threading.Thread] = []
        if lb_dev:
            t_lb = threading.Thread(
                target=_reader_worker, args=(lb_dev, latest_lb, "Loopback"), daemon=True
            )
            t_lb.start()
            threads.append(t_lb)
        if mic_dev:
            t_mic = threading.Thread(
                target=_reader_worker, args=(mic_dev, latest_mic, "Microphone"), daemon=True
            )
            t_mic.start()
            threads.append(t_mic)

        if not threads:
            self.is_recording = False
            return

        buffer: List[np.ndarray] = []
        current_samples = 0

        try:
            while not self._stop_event.is_set():
                time.sleep(0.25)
                with lock:
                    lb_chunks = latest_lb[:]
                    mic_chunks = latest_mic[:]
                    latest_lb.clear()
                    latest_mic.clear()

                lb_vec = np.concatenate(lb_chunks) if lb_chunks else np.zeros(0, dtype=np.float32)
                mic_vec = np.concatenate(mic_chunks) if mic_chunks else np.zeros(0, dtype=np.float32)

                if len(lb_vec) == 0 and len(mic_vec) == 0:
                    if self.level_callback:
                        self.level_callback(0.0)
                    continue

                if len(lb_vec) > 0 and len(mic_vec) > 0:
                    max_len = max(len(lb_vec), len(mic_vec))
                    if len(lb_vec) < max_len:
                        lb_vec = np.pad(lb_vec, (0, max_len - len(lb_vec)))
                    if len(mic_vec) < max_len:
                        mic_vec = np.pad(mic_vec, (0, max_len - len(mic_vec)))
                    block = np.clip(lb_vec + mic_vec, -1.0, 1.0).astype(np.float32)
                elif len(mic_vec) > 0:
                    block = mic_vec
                else:
                    block = lb_vec

                # Real-time 250ms RMS callback for responsive UI audio meter
                block_rms = float(np.sqrt(np.mean(block**2))) if len(block) > 0 else 0.0
                if self.level_callback:
                    self.level_callback(block_rms)

                buffer.append(block)
                current_samples += len(block)

                # Wait for either target_samples AND a brief natural pause at the end of the block,
                # or max 1.5x target_samples so sentences aren't cut in the middle of a word!
                tail_len = min(len(block), int(SAMPLE_RATE * 0.2))
                tail_rms = float(np.sqrt(np.mean(block[-tail_len:] ** 2))) if tail_len > 0 else 0.0
                is_pause_at_end = tail_rms < 0.012

                if (current_samples >= target_samples and is_pause_at_end) or (
                    current_samples >= int(target_samples * 1.45)
                ):
                    chunk = np.concatenate(buffer)
                    buffer = []
                    current_samples = 0
                    chunk_rms = float(np.sqrt(np.mean(chunk**2)))
                    self.chunk_callback(chunk, chunk_rms)
        except Exception as exc:
            print(f"[BackendLiveRecorder] Coordinator loop error: {exc}")
        finally:
            if buffer:
                chunk = np.concatenate(buffer)
                if len(chunk) > int(SAMPLE_RATE * 0.35):
                    chunk_rms = float(np.sqrt(np.mean(chunk**2)))
                    self.chunk_callback(chunk, chunk_rms)
            self.is_recording = False
