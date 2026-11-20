"""
Offline Model Manager for Wishper.
Handles downloading, caching, and verifying local AI models:
1. Sherpa-ONNX Whisper INT8 models (tiny, base, small, medium, turbo) for Speech-to-Text
2. Specialized OPUS-MT ONNX INT8 models (Danish <-> English, Spanish -> English, German -> English, etc.)
3. Meta NLLB-200-distilled-600M (ONNX INT8) for 200-language Any-to-Any Translation
4. Sherpa-ONNX 3D-Speaker ONNX model for Speaker Diarization / Voice Embeddings
"""
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, List

import config
from config import (
    DEFAULT_WHISPER_MODEL,
    NLLB_REPO_ID,
    OPUS_PAIR_REPOS,
    SPEAKER_EMBEDDING_MODEL_URL,
    TRANSLATION_MODELS,
    WHISPER_MODELS,
)

WHISPER_MAP = {m["id"]: m for m in WHISPER_MODELS}


def hf_hub_download(repo_id: str, filename: str, local_dir: str) -> str:
    """Download a public model file directly from Hugging Face via standard-library urllib (zero native DLL dependencies)."""
    dest = Path(local_dir) / filename
    if dest.exists() and dest.stat().st_size > 0:
        return str(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "WispAI/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp, open(tmp_path, "wb") as out_f:
        shutil.copyfileobj(resp, out_f, length=1024 * 1024)
    tmp_path.replace(dest)
    return str(dest)


def get_whisper_model_dir(model_id: str) -> Path:
    if model_id == "large-v3-turbo":
        model_id = "turbo"
    return config.MODELS_DIR / f"whisper-onnx-{model_id}"


def get_whisper_onnx_paths(model_id: str) -> Dict[str, Path]:
    if model_id == "large-v3-turbo":
        model_id = "turbo"
    info = WHISPER_MAP.get(model_id, WHISPER_MAP[DEFAULT_WHISPER_MODEL])
    prefix = info["prefix"]
    d = get_whisper_model_dir(model_id)
    return {
        "dir": d,
        "encoder": d / f"{prefix}-encoder.int8.onnx",
        "decoder": d / f"{prefix}-decoder.int8.onnx",
        "tokens": d / f"{prefix}-tokens.txt",
    }


def is_whisper_model_downloaded(model_id: str) -> bool:
    paths = get_whisper_onnx_paths(model_id)
    return (
        paths["encoder"].exists()
        and paths["decoder"].exists()
        and paths["tokens"].exists()
    )


def resolve_local_whisper_model(model_id: str = DEFAULT_WHISPER_MODEL) -> Dict[str, Path]:
    """
    Strictly local-only resolver used during transcription/language detection.
    Never makes any network calls. If `model_id` is not on disk, falls back to any
    other downloaded local Whisper model on disk.
    """
    if is_whisper_model_downloaded(model_id):
        return get_whisper_onnx_paths(model_id)
    for m in WHISPER_MODELS:
        if is_whisper_model_downloaded(m["id"]):
            return get_whisper_onnx_paths(m["id"])
    raise FileNotFoundError(
        f"Whisper model '{model_id}' is not downloaded locally. Please download a model in Settings."
    )


def is_speaker_model_downloaded() -> bool:
    return (
        config.SPEAKER_EMBEDDING_MODEL_PATH.exists()
        and config.SPEAKER_EMBEDDING_MODEL_PATH.stat().st_size > 1_000_000
    )


def get_opus_pair_dir(pair_key: str) -> Path:
    return config.MODELS_DIR / f"opus-mt-{pair_key}"


def is_opus_pair_downloaded(pair_key: str) -> bool:
    d = get_opus_pair_dir(pair_key)
    return (
        (d / "onnx" / "encoder_model_quantized.onnx").exists()
        and (d / "onnx" / "decoder_model_quantized.onnx").exists()
        and (d / "source.spm").exists()
        and (d / "vocab.json").exists()
    )


def is_opus_da_en_ready() -> bool:
    return is_opus_pair_downloaded("da-en") and is_opus_pair_downloaded("en-da")


def is_nllb_model_downloaded() -> bool:
    return (
        (config.NLLB_LOCAL_DIR / "onnx" / "encoder_model_quantized.onnx").exists()
        and (config.NLLB_LOCAL_DIR / "onnx" / "decoder_model_quantized.onnx").exists()
        and (config.NLLB_LOCAL_DIR / "tokenizer.json").exists()
    )


def ensure_opus_pair_model(pair_key: str = "da-en") -> Path:
    """Download a specialized Helsinki-NLP OPUS-MT ONNX INT8 model for `pair_key` (e.g., 'da-en', 'en-da', 'es-en')."""
    target_dir = get_opus_pair_dir(pair_key)
    if is_opus_pair_downloaded(pair_key):
        return target_dir

    repo_id = OPUS_PAIR_REPOS.get(pair_key)
    if not repo_id:
        raise ValueError(f"Unsupported OPUS language pair: {pair_key}")

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ModelManager] Downloading specialized OPUS-MT '{pair_key}' from {repo_id} to {target_dir} ...")
    files = [
        "config.json",
        "source.spm",
        "target.spm",
        "vocab.json",
        "onnx/encoder_model_quantized.onnx",
        "onnx/decoder_model_quantized.onnx",
    ]
    for f in files:
        hf_hub_download(repo_id=repo_id, filename=f, local_dir=str(target_dir))
    return target_dir


def ensure_whisper_model(model_id: str = DEFAULT_WHISPER_MODEL) -> Dict[str, Path]:
    """Ensure the specified Whisper ONNX INT8 model is stored locally."""
    if model_id == "large-v3-turbo":
        model_id = "turbo"
    if model_id not in WHISPER_MAP:
        model_id = DEFAULT_WHISPER_MODEL

    paths = get_whisper_onnx_paths(model_id)
    if is_whisper_model_downloaded(model_id):
        return paths

    info = WHISPER_MAP[model_id]
    repo_id = info["hf_repo"]
    prefix = info["prefix"]
    target_dir = paths["dir"]
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ModelManager] Downloading Whisper ONNX '{model_id}' from {repo_id} to {target_dir} ...")
    files_to_fetch = [
        f"{prefix}-encoder.int8.onnx",
        f"{prefix}-decoder.int8.onnx",
        f"{prefix}-tokens.txt",
    ]
    for fname in files_to_fetch:
        hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            local_dir=str(target_dir),
        )
    return paths


def ensure_speaker_model() -> Path:
    """Ensure the ONNX speaker embedding model is stored locally."""
    if is_speaker_model_downloaded():
        return config.SPEAKER_EMBEDDING_MODEL_PATH

    config.SPEAKER_EMBEDDING_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[ModelManager] Downloading Speaker Embedding ONNX model to {config.SPEAKER_EMBEDDING_MODEL_PATH} ..."
    )
    tmp_path = config.SPEAKER_EMBEDDING_MODEL_PATH.with_suffix(".tmp")
    urllib.request.urlretrieve(SPEAKER_EMBEDDING_MODEL_URL, str(tmp_path))
    tmp_path.replace(config.SPEAKER_EMBEDDING_MODEL_PATH)
    return config.SPEAKER_EMBEDDING_MODEL_PATH


def ensure_nllb_model() -> Path:
    """Ensure the NLLB-200-distilled-600M ONNX INT8 model is stored locally."""
    if is_nllb_model_downloaded():
        return config.NLLB_LOCAL_DIR

    config.NLLB_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[ModelManager] Downloading NLLB-200 Offline ONNX Translator to {config.NLLB_LOCAL_DIR} ...")
    files_to_fetch = [
        "tokenizer.json",
        "tokenizer_config.json",
        "config.json",
        "onnx/encoder_model_quantized.onnx",
        "onnx/decoder_model_quantized.onnx",
    ]
    for fname in files_to_fetch:
        hf_hub_download(
            repo_id=NLLB_REPO_ID,
            filename=fname,
            local_dir=str(config.NLLB_LOCAL_DIR),
        )
    return config.NLLB_LOCAL_DIR


def delete_local_model(model_type: str, model_id: str = "") -> None:
    """Delete a downloaded local AI model from disk to free up storage."""
    if model_type == "whisper":
        target_dir = get_whisper_model_dir(model_id)
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
    elif model_type in ("opus", "opus-mt-da-en") or (
        model_type == "translation" and model_id == "opus-mt-da-en"
    ):
        for pair in ("da-en", "en-da"):
            d = get_opus_pair_dir(pair)
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
    elif model_type in ("nllb", "nllb-200-600M") or (
        model_type == "translation" and model_id == "nllb-200-600M"
    ):
        if config.NLLB_LOCAL_DIR.exists():
            shutil.rmtree(config.NLLB_LOCAL_DIR, ignore_errors=True)
    elif model_type == "speaker":
        if config.SPEAKER_EMBEDDING_MODEL_PATH.exists():
            config.SPEAKER_EMBEDDING_MODEL_PATH.unlink(missing_ok=True)


def get_models_status() -> Dict:
    """Return readiness status of all local AI models for the UI."""
    whisper_list: List[Dict] = []
    for m in WHISPER_MODELS:
        whisper_list.append(
            {
                **m,
                "downloaded": is_whisper_model_downloaded(m["id"]),
                "local_path": str(get_whisper_model_dir(m["id"])),
            }
        )

    translation_list: List[Dict] = []
    for tm in TRANSLATION_MODELS:
        if tm["id"] == "opus-mt-da-en":
            dl = is_opus_da_en_ready()
            lpath = str(get_opus_pair_dir("da-en"))
        elif tm["id"] == "nllb-200-600M":
            dl = is_nllb_model_downloaded()
            lpath = str(config.NLLB_LOCAL_DIR)
        else:
            dl = True
            lpath = "Built into Whisper ASR"
        translation_list.append({**tm, "downloaded": dl, "local_path": lpath})

    spk_ready = is_speaker_model_downloaded()
    any_whisper_ready = any(w["downloaded"] for w in whisper_list)
    any_neural_translator_ready = is_opus_da_en_ready() or is_nllb_model_downloaded()

    return {
        "storage_directory": str(config.DATA_DIR),
        "models_directory": str(config.MODELS_DIR),
        "sessions_directory": str(config.SESSIONS_DIR),
        "needs_onboarding": not (spk_ready and any_whisper_ready and any_neural_translator_ready),
        "whisper_models": whisper_list,
        "translation_models": translation_list,
        "speaker_diarization": {
            "id": "3dspeaker-eres2net",
            "name": "3D-Speaker Voice Embedding (ONNX)",
            "size_mb": "26",
            "downloaded": spk_ready,
            "local_path": str(config.SPEAKER_EMBEDDING_MODEL_PATH),
        },
        "translator": {
            "id": "opus-mt-da-en",
            "name": "OPUS-MT Specialized Danish ↔ English + NLLB-200",
            "size_mb": "85",
            "downloaded": any_neural_translator_ready,
            "local_path": str(get_opus_pair_dir("da-en")),
        },
    }
