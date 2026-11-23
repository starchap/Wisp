"""
FastAPI + WebSocket Server for Wishper Local AI Engine.
100% offline-capable transcription, speaker diarization, per-speaker language assignment,
retroactive re-transcription, and single-target-language translation.
"""
import asyncio
import io
import json
import threading
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from audio_capture import BackendLiveRecorder, list_audio_devices
import config
from config import (
    CLARIFICATION_MODELS,
    DEFAULT_CLARIFICATION_MODEL,
    DEFAULT_TRANSLATION_MODEL,
    DEFAULT_WHISPER_MODEL,
    LANGUAGE_NAME_MAP,
    SUPPORTED_LANGUAGES,
    TRANSLATION_MODELS,
    WHISPER_MODELS,
    set_storage_path,
)
from models_manager import (
    delete_local_model,
    ensure_clarification_model,
    ensure_nllb_model,
    ensure_opus_pair_model,
    ensure_speaker_model,
    ensure_whisper_model,
    get_models_status,
)
from session_store import (
    SAMPLE_RATE,
    SessionStore,
    decode_audio_to_16k_mono,
)
from transcription import LocalSpeechPipeline

app = FastAPI(
    title="Wishper Local AI Backend",
    description="100% Local AI Multi-Speaker Meeting Transcriber & Translator",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = SessionStore()
pipeline = LocalSpeechPipeline(store)

# Track active WebSocket connections for live updates
active_websockets: List[WebSocket] = []
main_event_loop: Optional[asyncio.AbstractEventLoop] = None

# Active backend live recorder state
live_state = {
    "is_recording": False,
    "session_id": None,
    "loopback_id": None,
    "mic_id": None,
    "rms_level": 0.0,
    "active_speaker_id": None,
}
backend_recorder: Optional[BackendLiveRecorder] = None


def broadcast_event(event_type: str, payload: dict) -> None:
    """Thread-safe broadcast to all connected frontend WebSocket clients."""
    if not main_event_loop or not active_websockets:
        return
    message = json.dumps({"type": event_type, "data": payload}, ensure_ascii=False)

    async def _send_all():
        dead = []
        for ws in active_websockets:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in active_websockets:
                active_websockets.remove(ws)

    asyncio.run_coroutine_threadsafe(_send_all(), main_event_loop)


@app.on_event("startup")
async def on_startup():
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()


# ---------------------------------------------------------------------------
# System & Model Management Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/config")
def get_config():
    return {
        "languages": SUPPORTED_LANGUAGES,
        "whisper_models": WHISPER_MODELS,
        "default_whisper_model": DEFAULT_WHISPER_MODEL,
        "clarification_models": CLARIFICATION_MODELS,
        "default_clarification_model": DEFAULT_CLARIFICATION_MODEL,
        "translation_models": TRANSLATION_MODELS,
        "default_translation_model": DEFAULT_TRANSLATION_MODEL,
        "device": pipeline.device,
        "compute_type": pipeline.compute_type,
        "models_status": get_models_status(),
        "audio_devices": list_audio_devices(),
        "live_state": live_state,
    }


@app.get("/api/models/status")
def api_models_status():
    return get_models_status()


class StoragePathRequest(BaseModel):
    storage_path: str


@app.post("/api/settings/storage")
def api_set_storage_path(req: StoragePathRequest):
    """Change the root local directory where Wisp stores AI models and session recordings."""
    if not req.storage_path or not req.storage_path.strip():
        raise HTTPException(status_code=400, detail="Storage path cannot be empty")
    try:
        new_root = set_storage_path(req.storage_path)
        store.reload_from_storage()
        status = get_models_status()
        broadcast_event(
            "model_progress",
            {
                "status": f"Storage path updated to {new_root}",
                "busy": False,
                "models_status": status,
            },
        )
        return status
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid storage path: {exc}")


class DownloadModelRequest(BaseModel):
    model_type: str  # "whisper", "speaker", "clarification", "opus", "nllb", "all_recommended", or "onboarding"
    whisper_id: Optional[str] = DEFAULT_WHISPER_MODEL
    clarification_id: Optional[str] = DEFAULT_CLARIFICATION_MODEL
    translation_id: Optional[str] = DEFAULT_TRANSLATION_MODEL


@app.post("/api/models/download")
def api_download_model(req: DownloadModelRequest):
    """Download and cache requested local AI models for 100% offline operation."""
    def _worker():
        try:
            if req.model_type in ("speaker", "all_recommended", "onboarding"):
                broadcast_event("model_progress", {"status": "Downloading Step 1/4: 3D-Speaker Diarization ONNX Model (~26 MB)...", "busy": True})
                ensure_speaker_model()
            if req.model_type in ("whisper", "all_recommended", "onboarding"):
                w_id = req.whisper_id or DEFAULT_WHISPER_MODEL
                broadcast_event("model_progress", {"status": f"Downloading Step 2/4: Whisper '{w_id}' Speech-to-Text Model...", "busy": True})
                ensure_whisper_model(w_id)
            if req.model_type == "clarification" or (
                req.model_type == "onboarding"
                and req.clarification_id
                and req.clarification_id not in ("none", "whisper-context")
            ):
                c_id = req.clarification_id or DEFAULT_CLARIFICATION_MODEL
                broadcast_event("model_progress", {"status": f"Downloading Step 3/4: Context Clarifier '{c_id}' ONNX Model...", "busy": True})
                ensure_clarification_model(c_id)
            if req.model_type in ("opus", "all_recommended") or (
                req.model_type == "onboarding" and req.translation_id == "opus-mt-da-en"
            ):
                broadcast_event("model_progress", {"status": "Downloading Step 4/4: Specialized OPUS-MT Danish <-> English ONNX Models (~85 MB)...", "busy": True})
                ensure_opus_pair_model("da-en")
                ensure_opus_pair_model("en-da")
            if req.model_type == "nllb" or (
                req.model_type == "onboarding" and req.translation_id == "nllb-200-600M"
            ):
                broadcast_event("model_progress", {"status": "Downloading Step 4/4: Meta NLLB-200 Offline Translator (~580 MB)...", "busy": True})
                ensure_nllb_model()
            broadcast_event("model_progress", {"status": "All requested models are ready offline!", "busy": False, "models_status": get_models_status()})
        except Exception as exc:
            broadcast_event("model_progress", {"status": f"Download error: {exc}", "busy": False, "error": str(exc)})

    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "started"}


class DeleteModelRequest(BaseModel):
    model_type: str  # "whisper", "speaker", "clarification", "opus", "nllb", or "translation"
    model_id: Optional[str] = ""


@app.post("/api/models/delete")
def api_delete_model(req: DeleteModelRequest):
    """Delete a downloaded local AI model from disk (only triggered from Settings)."""
    try:
        pipeline.evict_model_cache(req.model_type, req.model_id or "")
        delete_local_model(req.model_type, req.model_id or "")
        status = get_models_status()
        broadcast_event(
            "model_progress",
            {
                "status": f"Deleted model '{req.model_id or req.model_type}' from local storage.",
                "busy": False,
                "models_status": status,
            },
        )
        return status
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not delete model: {exc}")


@app.get("/api/devices")
def api_list_devices():
    return list_audio_devices()


# ---------------------------------------------------------------------------
# Session CRUD & Audio Backtracking Endpoints
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    title: str = "New Meeting"
    session_type: str = "file"
    target_language: str = "en"
    whisper_model: str = DEFAULT_WHISPER_MODEL
    clarification_model: str = DEFAULT_CLARIFICATION_MODEL
    translation_model: str = DEFAULT_TRANSLATION_MODEL
    spoken_languages: Optional[List[str]] = None


@app.get("/api/sessions")
def api_list_sessions():
    return {"sessions": store.list_sessions()}


@app.post("/api/sessions")
def api_create_session(req: CreateSessionRequest):
    session = store.create_session(
        title=req.title,
        session_type=req.session_type,
        target_language=req.target_language,
        whisper_model=req.whisper_model,
        clarification_model=req.clarification_model,
        translation_model=req.translation_model,
        spoken_languages=req.spoken_languages,
    )
    return session.to_dict()


@app.get("/api/sessions/{session_id}")
def api_get_session(session_id: str):
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.to_dict()


@app.delete("/api/sessions/{session_id}")
def api_delete_session(session_id: str):
    if live_state["is_recording"] and live_state["session_id"] == session_id:
        api_stop_live_recording()
    ok = store.delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True}


class UpdateSessionRequest(BaseModel):
    title: Optional[str] = None
    target_language: Optional[str] = None
    whisper_model: Optional[str] = None
    clarification_model: Optional[str] = None
    translation_model: Optional[str] = None
    spoken_languages: Optional[List[str]] = None
    retranslate_existing: bool = True
    retranscribe_existing: bool = False


def _trigger_session_reprocess_async(session_id: str) -> None:
    session = store.get_session(session_id)
    if not session:
        return

    def _progress(msg: str, pct: int):
        broadcast_event(
            "upload_progress",
            {"session_id": session_id, "message": msg, "percent": pct},
        )

    def _worker():
        try:
            pipeline.retranscribe_and_translate_all_segments(
                session, progress_cb=_progress
            )
            broadcast_event("session_updated", session.to_dict())
        except Exception as exc:
            broadcast_event(
                "upload_progress",
                {
                    "session_id": session_id,
                    "message": f"Reprocess error: {exc}",
                    "percent": 100,
                    "error": str(exc),
                },
            )

    threading.Thread(target=_worker, daemon=True).start()


@app.patch("/api/sessions/{session_id}")
def api_update_session(session_id: str, req: UpdateSessionRequest):
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    retranslate_needed = False
    clarification_changed = False
    whisper_changed = False
    spoken_langs_changed = False
    if req.title is not None:
        session.title = req.title
    if req.whisper_model is not None and req.whisper_model != session.whisper_model:
        session.whisper_model = req.whisper_model
        whisper_changed = True
    if (
        req.clarification_model is not None
        and req.clarification_model != session.clarification_model
    ):
        session.clarification_model = req.clarification_model
        clarification_changed = True
    if req.translation_model is not None and req.translation_model != session.translation_model:
        session.translation_model = req.translation_model
        retranslate_needed = True
    if req.target_language is not None and req.target_language != session.target_language:
        session.target_language = req.target_language
        retranslate_needed = True
    if req.spoken_languages is not None:
        clean_spoken = [
            c for c in req.spoken_languages if c and c not in ("auto", "none")
        ]
        if clean_spoken != session.spoken_languages:
            session.spoken_languages = clean_spoken
            spoken_langs_changed = True

    store.save_session(session)

    if (req.retranscribe_existing or whisper_changed or spoken_langs_changed) and (
        session.segments or store.get_audio_path(session_id).exists()
    ):
        _trigger_session_reprocess_async(session_id)
    elif clarification_changed and session.segments:
        pipeline.reclarify_and_retranslate_all_segments(session)
    elif retranslate_needed and req.retranslate_existing and session.segments:
        pipeline.retranslate_all_segments(session)

    broadcast_event("session_updated", session.to_dict())
    return session.to_dict()


@app.post("/api/sessions/{session_id}/reprocess")
def api_reprocess_session(session_id: str):
    """Re-run Speech-to-Text transcription and translation on a session's recorded audio."""
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _trigger_session_reprocess_async(session_id)
    return {"status": "reprocessing", "session_id": session_id}


@app.get("/api/sessions/{session_id}/audio")
def api_get_session_audio(session_id: str, download: int = 0):
    """Serve the recorded WAV file for full playback, segment backtracking, or direct download."""
    session = store.get_session(session_id)
    audio_path = store.get_audio_path(session_id)
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio recording not found")

    safe_title = f"WispAI-{session_id}"
    if session and session.title:
        cleaned = "".join(c if (c.isalnum() or c in " -_") else "_" for c in session.title).strip()
        if cleaned:
            safe_title = cleaned.replace(" ", "_")

    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{safe_title}.wav"'

    return FileResponse(
        path=str(audio_path),
        media_type="audio/wav",
        filename=f"{safe_title}.wav",
        headers=headers,
    )


@app.post("/api/sessions/{session_id}/open-folder")
def api_open_session_folder(session_id: str):
    """Open the native OS file manager (Windows Explorer, macOS Finder, Linux) at the session's audio file."""
    import os
    import subprocess
    import sys

    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    audio_path = store.get_audio_path(session_id).resolve()
    session_dir = audio_path.parent.resolve()
    session_dir.mkdir(parents=True, exist_ok=True)

    target_to_select = audio_path if audio_path.exists() else session_dir

    try:
        if sys.platform == "win32":
            if audio_path.exists():
                subprocess.Popen(["explorer", f"/select,{str(audio_path)}"])
            else:
                os.startfile(str(session_dir))  # type: ignore
        elif sys.platform == "darwin":
            if audio_path.exists():
                subprocess.Popen(["open", "-R", str(audio_path)])
            else:
                subprocess.Popen(["open", str(session_dir)])
        else:
            subprocess.Popen(["xdg-open", str(session_dir)])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not open folder: {exc}")

    return {"opened": True, "path": str(target_to_select)}


# ---------------------------------------------------------------------------
# Speaker Management & Per-Speaker Language Assignment / Retroactive Reprocess
# ---------------------------------------------------------------------------

class UpdateSpeakerRequest(BaseModel):
    name: Optional[str] = None
    language: Optional[str] = None
    color: Optional[str] = None
    reprocess_segments: bool = True


@app.patch("/api/sessions/{session_id}/speakers/{speaker_id}")
def api_update_speaker(session_id: str, speaker_id: str, req: UpdateSpeakerRequest):
    """
    Rename a speaker (e.g. Speaker 1 -> 'Anders') or change their spoken language
    (e.g. Auto -> Danish 'da' or Spanish 'es').
    When `language` changes and `reprocess_segments=True`, retroactively re-transcribes
    all audio segments spoken by this speaker in that language and translates to the output language!
    """
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    spk = store.ensure_speaker(session, speaker_id)
    lang_changed = False

    if req.name is not None and req.name.strip():
        spk.name = req.name.strip()
    if req.color is not None:
        spk.color = req.color
    if req.language is not None and req.language != spk.language:
        spk.language = req.language
        lang_changed = True

    store.save_session(session)

    if lang_changed and req.reprocess_segments:
        # Mark speaker segments as reprocessing and run in background so UI stays snappy
        def _reprocess():
            broadcast_event(
                "speaker_reprocessing",
                {"session_id": session_id, "speaker_id": speaker_id, "status": "running"},
            )
            pipeline.reprocess_speaker_segments(session, speaker_id)
            broadcast_event("session_updated", session.to_dict())
            broadcast_event(
                "speaker_reprocessing",
                {"session_id": session_id, "speaker_id": speaker_id, "status": "completed"},
            )

        threading.Thread(target=_reprocess, daemon=True).start()

    return session.to_dict()


class CreateSpeakerRequest(BaseModel):
    name: Optional[str] = None
    language: str = "auto"


@app.post("/api/sessions/{session_id}/speakers")
def api_add_speaker(session_id: str, req: CreateSpeakerRequest):
    """Pre-configure a speaker before or during a meeting (e.g., Speaker 1: Danish, Speaker 2: Spanish)."""
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    next_idx = len(session.speakers) + 1
    spk_id = f"speaker_{next_idx}"
    while spk_id in session.speakers:
        next_idx += 1
        spk_id = f"speaker_{next_idx}"

    spk = store.ensure_speaker(session, spk_id, default_lang=req.language)
    if req.name and req.name.strip():
        spk.name = req.name.strip()
    store.save_session(session)
    broadcast_event("session_updated", session.to_dict())
    return session.to_dict()


class MergeSpeakersRequest(BaseModel):
    source_speaker_id: str
    target_speaker_id: str
    reprocess_with_target_language: bool = True


@app.post("/api/sessions/{session_id}/speakers/merge")
def api_merge_speakers(session_id: str, req: MergeSpeakersRequest):
    """
    Merge `source_speaker_id` into `target_speaker_id`:
    1. Reassign all transcript segments from source -> target
    2. Combine & normalize their voice embedding centroids so future live audio recognizes the merged voice
    3. Remove `source_speaker_id` from `session.speakers`
    4. Optionally re-transcribe the newly merged segments if `target_speaker_id` has a specific language assigned
    """
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    src_id = req.source_speaker_id
    tgt_id = req.target_speaker_id
    if src_id == tgt_id:
        raise HTTPException(status_code=400, detail="Cannot merge a speaker with itself")
    if src_id not in session.speakers or tgt_id not in session.speakers:
        raise HTTPException(status_code=404, detail="Speaker not found in session")

    src_spk = session.speakers[src_id]
    tgt_spk = session.speakers[tgt_id]

    # Combine voice embeddings weighted by segment counts so future live chunks match the merged voice
    if src_id in session.embeddings and tgt_id in session.embeddings:
        emb_src = np.array(session.embeddings[src_id], dtype=np.float32)
        emb_tgt = np.array(session.embeddings[tgt_id], dtype=np.float32)
        if emb_src.shape == emb_tgt.shape:
            w_src = max(1, src_spk.segment_count)
            w_tgt = max(1, tgt_spk.segment_count)
            merged_emb = (w_src * emb_src + w_tgt * emb_tgt) / float(w_src + w_tgt)
            norm = np.linalg.norm(merged_emb)
            if norm > 1e-8:
                merged_emb /= norm
            session.embeddings[tgt_id] = merged_emb.tolist()
    elif src_id in session.embeddings and tgt_id not in session.embeddings:
        session.embeddings[tgt_id] = session.embeddings[src_id]

    session.embeddings.pop(src_id, None)

    # Reassign all segments from source -> target
    merged_segments = []
    for seg in session.segments:
        if seg.speaker_id == src_id:
            seg.speaker_id = tgt_id
            merged_segments.append(seg)

    # Delete source speaker profile
    del session.speakers[src_id]
    store.save_session(session)
    broadcast_event("session_updated", session.to_dict())

    # If target speaker has a specific language (e.g. Danish or Spanish) and source segments differed, reprocess
    if (
        req.reprocess_with_target_language
        and tgt_spk.language != "auto"
        and any(s.source_language != tgt_spk.language for s in merged_segments)
    ):
        def _reprocess_merged():
            broadcast_event(
                "speaker_reprocessing",
                {"session_id": session_id, "speaker_id": tgt_id, "status": "running"},
            )
            for seg in merged_segments:
                if seg.source_language == tgt_spk.language:
                    continue
                clip = store.read_audio_slice(session.id, seg.start, seg.end)
                if len(clip) >= int(SAMPLE_RATE * 0.25):
                    conv_ctx, spk_ctx = pipeline._build_conversation_context(
                        session, current_speaker_id=tgt_id
                    )
                    det_lang, orig, trans, conf, raw_stt = pipeline.transcribe_and_translate_clip(
                        clip,
                        forced_language=tgt_spk.language,
                        target_language=session.target_language,
                        model_id=session.whisper_model,
                        clarification_model=session.clarification_model,
                        translation_model=session.translation_model,
                        allowed_languages=session.spoken_languages,
                        conversation_context=conv_ctx,
                        speaker_context=spk_ctx,
                    )
                    if orig:
                        seg.source_language = det_lang
                        seg.original_text = orig
                        seg.translated_text = trans
                        seg.target_language = session.target_language
                        seg.confidence = round(conf, 3)
                        seg.raw_text = raw_stt
            store.save_session(session)
            broadcast_event("session_updated", session.to_dict())
            broadcast_event(
                "speaker_reprocessing",
                {"session_id": session_id, "speaker_id": tgt_id, "status": "completed"},
            )

        threading.Thread(target=_reprocess_merged, daemon=True).start()

    return session.to_dict()


class ReassignSegmentSpeakerRequest(BaseModel):
    speaker_id: str
    reprocess_with_speaker_language: bool = True


@app.patch("/api/sessions/{session_id}/segments/{segment_id}")
def api_reassign_segment_speaker(
    session_id: str, segment_id: str, req: ReassignSegmentSpeakerRequest
):
    """Allow manual correction if a user wants to reassign a specific line to another speaker."""
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    spk = store.ensure_speaker(session, req.speaker_id)
    target_seg = next((s for s in session.segments if s.id == segment_id), None)
    if not target_seg:
        raise HTTPException(status_code=404, detail="Segment not found")

    target_seg.speaker_id = spk.id
    if req.reprocess_with_speaker_language and spk.language != "auto":
        clip = store.read_audio_slice(session.id, target_seg.start, target_seg.end)
        if len(clip) > int(SAMPLE_RATE * 0.2):
            conv_ctx, spk_ctx = pipeline._build_conversation_context(
                session, current_speaker_id=spk.id
            )
            det_lang, orig, trans, conf, raw_stt = pipeline.transcribe_and_translate_clip(
                clip,
                forced_language=spk.language,
                target_language=session.target_language,
                model_id=session.whisper_model,
                clarification_model=session.clarification_model,
                translation_model=session.translation_model,
                allowed_languages=session.spoken_languages,
                conversation_context=conv_ctx,
                speaker_context=spk_ctx,
            )
            if orig:
                target_seg.source_language = det_lang
                target_seg.original_text = orig
                target_seg.translated_text = trans
                target_seg.target_language = session.target_language
                target_seg.confidence = round(conf, 3)
                target_seg.raw_text = raw_stt

    store.save_session(session)
    broadcast_event("session_updated", session.to_dict())
    return session.to_dict()


# ---------------------------------------------------------------------------
# Audio File Upload & Batch Diarization + Transcription + Translation
# ---------------------------------------------------------------------------

@app.post("/api/sessions/{session_id}/upload")
async def api_upload_recording(
    session_id: str,
    file: UploadFile = File(...),
    num_speakers: Optional[int] = Form(None),
    original_filename: Optional[str] = Form(None),
):
    """
    Import an existing meeting recording (.wav, .mp3, .m4a, .mp4, .flac, .ogg, .webm)
    and run local speaker diarization, per-speaker language transcription, and translation.
    """
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    raw_bytes = await file.read()
    display_filename = original_filename or file.filename or "audio.wav"
    try:
        audio_16k = decode_audio_to_16k_mono(raw_bytes, filename=file.filename or display_filename)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode audio file: {exc}")

    if len(audio_16k) == 0:
        raise HTTPException(status_code=400, detail="Audio file contains no valid audio samples.")

    session.source_file_name = display_filename
    session.speakers = {}
    session.segments = []
    session.embeddings = {}
    store.save_session(session)
    broadcast_event("session_updated", session.to_dict())

    def _progress(msg: str, pct: int):
        broadcast_event(
            "upload_progress",
            {"session_id": session_id, "message": msg, "percent": pct},
        )

    def _worker():
        try:
            pipeline.process_uploaded_file(
                session,
                audio_16k,
                num_speakers=num_speakers if (num_speakers and num_speakers > 0) else None,
                progress_cb=_progress,
            )
            broadcast_event("session_updated", session.to_dict())
        except Exception as exc:
            broadcast_event(
                "upload_progress",
                {"session_id": session_id, "message": f"Error: {exc}", "percent": 100, "error": str(exc)},
            )

    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "processing", "session_id": session_id}


# ---------------------------------------------------------------------------
# Live Meeting Recording (System Loopback + Mic)
# ---------------------------------------------------------------------------

class StartLiveRequest(BaseModel):
    session_id: str
    loopback_id: Optional[str] = None
    mic_id: Optional[str] = None
    chunk_seconds: float = 4.0


@app.post("/api/live/start")
def api_start_live_recording(req: StartLiveRequest):
    global backend_recorder
    session = store.get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if live_state["is_recording"]:
        api_stop_live_recording()

    last_speaker_ts = [0.0]

    def _on_level(rms: float):
        live_state["rms_level"] = round(rms, 4)
        if rms < 0.004 and (time.time() - last_speaker_ts[0]) > 2.2:
            live_state["active_speaker_id"] = None
        broadcast_event(
            "live_level",
            {
                "session_id": req.session_id,
                "rms": live_state["rms_level"],
                "duration": round(session.duration, 1),
                "active_speaker_id": live_state.get("active_speaker_id"),
            },
        )

    def _on_speaker_active(spk_id: str):
        last_speaker_ts[0] = time.time()
        live_state["active_speaker_id"] = spk_id
        broadcast_event(
            "live_speaker",
            {"session_id": req.session_id, "speaker_id": spk_id},
        )

    def _on_audio_chunk(chunk_16k: np.ndarray, rms: float):
        live_state["rms_level"] = round(rms, 4)
        new_segs = pipeline.process_live_chunk(
            session, chunk_16k, on_speaker_active=_on_speaker_active
        )
        if new_segs:
            last_speaker_ts[0] = time.time()
            live_state["active_speaker_id"] = new_segs[-1].speaker_id
            broadcast_event("session_updated", session.to_dict())
            broadcast_event(
                "live_speaker",
                {"session_id": req.session_id, "speaker_id": new_segs[-1].speaker_id},
            )

    backend_recorder = BackendLiveRecorder(
        chunk_callback=_on_audio_chunk,
        level_callback=_on_level,
        chunk_seconds=max(2.5, min(10.0, req.chunk_seconds)),
    )
    backend_recorder.start(loopback_id=req.loopback_id, mic_id=req.mic_id)

    live_state["is_recording"] = True
    live_state["session_id"] = req.session_id
    live_state["loopback_id"] = req.loopback_id
    live_state["mic_id"] = req.mic_id
    live_state["active_speaker_id"] = None
    broadcast_event("live_state", live_state)
    return live_state


@app.post("/api/live/stop")
def api_stop_live_recording():
    global backend_recorder
    if backend_recorder:
        backend_recorder.stop()
        backend_recorder = None

    live_state["is_recording"] = False
    live_state["rms_level"] = 0.0
    live_state["active_speaker_id"] = None
    broadcast_event("live_state", live_state)
    return live_state


# ---------------------------------------------------------------------------
# Export Plain Text / Markdown / SRT
# ---------------------------------------------------------------------------

def _fmt_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


@app.get("/api/sessions/{session_id}/export", response_class=PlainTextResponse)
def api_export_transcript(
    session_id: str,
    mode: str = "translated",  # "translated", "dual", "original"
    format: str = "clean",     # "clean" (Word copy-paste), "txt" (with timestamps), "md"
):
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if format == "clean":
        # Group consecutive segments from the same speaker into clean paragraphs:
        # Speaker 1:
        # Paragraph text...
        blocks: List[str] = []
        grouped: List[Tuple[str, List[str], List[str]]] = []

        for seg in session.segments:
            spk = session.speakers.get(seg.speaker_id)
            spk_name = spk.name if spk else seg.speaker_id
            orig_t = seg.original_text.strip()
            trans_t = seg.translated_text.strip()
            if grouped and grouped[-1][0] == spk_name:
                if orig_t:
                    grouped[-1][1].append(orig_t)
                if trans_t:
                    grouped[-1][2].append(trans_t)
            else:
                grouped.append(
                    (
                        spk_name,
                        [orig_t] if orig_t else [],
                        [trans_t] if trans_t else [],
                    )
                )

        for spk_name, orig_list, trans_list in grouped:
            orig_joined = " ".join(orig_list).strip()
            trans_joined = " ".join(trans_list).strip()
            if mode == "original":
                blocks.append(f"{spk_name}:\n{orig_joined}")
            elif mode == "dual" and orig_joined.lower() != trans_joined.lower():
                blocks.append(f"{spk_name}:\n{trans_joined}\n({orig_joined})")
            else:
                blocks.append(f"{spk_name}:\n{trans_joined}")

        return "\n\n\n".join(blocks)

    target_lang_name = LANGUAGE_NAME_MAP.get(session.target_language, session.target_language)
    lines: List[str] = []

    if format == "md":
        lines.append(f"# {session.title}")
        lines.append(f"**Output Language:** {target_lang_name}  ")
        lines.append(f"**Speakers:** {', '.join(s.name for s in session.speakers.values())}")
        lines.append("")
        lines.append("---")
        lines.append("")
    else:
        lines.append(f"MEETING: {session.title}")
        lines.append(f"OUTPUT LANGUAGE: {target_lang_name}")
        lines.append("=" * 60)
        lines.append("")

    for seg in session.segments:
        spk = session.speakers.get(seg.speaker_id)
        spk_name = spk.name if spk else seg.speaker_id
        ts = f"[{_fmt_timestamp(seg.start)} - {_fmt_timestamp(seg.end)}]"
        src_lang_label = seg.source_language.upper()
        tgt_lang_label = seg.target_language.upper()

        if mode == "translated":
            lines.append(f"{ts} {spk_name} ({src_lang_label} -> {tgt_lang_label}): {seg.translated_text}")
        elif mode == "original":
            lines.append(f"{ts} {spk_name} ({src_lang_label}): {seg.original_text}")
        else:
            # Dual mode: both original and translated
            lines.append(f"{ts} {spk_name} [{src_lang_label} -> {tgt_lang_label}]:")
            lines.append(f"  Original ({src_lang_label}):   {seg.original_text}")
            lines.append(f"  Translated ({tgt_lang_label}): {seg.translated_text}")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# WebSocket Endpoint for Real-Time Events & Browser Audio Chunk Streaming
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            msg = await websocket.receive()
            if "text" in msg and msg["text"]:
                data = json.loads(msg["text"])
                if data.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            elif "bytes" in msg and msg["bytes"]:
                # Optional direct PCM float32 chunk from Electron/Browser MediaRecorder/AudioContext
                raw_bytes = msg["bytes"]
                session_id = live_state.get("session_id")
                if session_id:
                    session = store.get_session(session_id)
                    if session:
                        audio_16k = np.frombuffer(raw_bytes, dtype=np.float32)
                        await asyncio.to_thread(pipeline.process_live_chunk, session, audio_16k)
                        broadcast_event("session_updated", session.to_dict())
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in active_websockets:
            active_websockets.remove(websocket)


if __name__ == "__main__":
    import multiprocessing
    import os
    import sys
    import uvicorn

    multiprocessing.freeze_support()

    # When running as a bundled PyInstaller executable without an attached console,
    # ensure sys.stdout and sys.stderr are valid streams so uvicorn logging never fails.
    if sys.stdout is None or sys.stderr is None:
        log_path = config.BASE_DIR / "backend.log"
        log_stream = open(log_path, "a", encoding="utf-8", buffering=1)
        if sys.stdout is None:
            sys.stdout = log_stream
        if sys.stderr is None:
            sys.stderr = log_stream

    # Pass the `app` instance directly (not `"main:app"`) so PyInstaller frozen binaries
    # do not fail trying to dynamically import `"main"`.
    uvicorn.run(app, host="127.0.0.1", port=8765, reload=False, log_config=None)
