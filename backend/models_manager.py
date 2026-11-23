"""
Offline Model Manager for Wishper.
Handles downloading, caching, and verifying local AI models:
1. Sherpa-ONNX Whisper INT8 models (tiny, base, small, medium, turbo) for Speech-to-Text
2. Specialized OPUS-MT ONNX INT8 models (Danish <-> English, Spanish -> English, German -> English, etc.)
3. Meta NLLB-200-distilled-600M (ONNX INT8) for 200-language Any-to-Any Translation
4. Sherpa-ONNX 3D-Speaker ONNX model for Speaker Diarization / Voice Embeddings
"""
import shutil
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

_PROGRESS_HOOK: Optional[Callable[[str], None]] = None


def set_download_progress_hook(hook: Optional[Callable[[str], None]]) -> None:
    global _PROGRESS_HOOK
    _PROGRESS_HOOK = hook

try:
    import certifi
    HAS_CERTIFI = True
except ImportError:
    HAS_CERTIFI = False

import config
from config import (
    CLARIFICATION_MODELS,
    DEFAULT_CLARIFICATION_MODEL,
    DEFAULT_WHISPER_MODEL,
    NLLB_REPO_ID,
    OPUS_PAIR_REPOS,
    SPEAKER_EMBEDDING_MODEL_URL,
    TRANSLATION_MODELS,
    WHISPER_MODELS,
)

WHISPER_MAP = {m["id"]: m for m in WHISPER_MODELS}
CLARIFICATION_MAP = {m["id"]: m for m in CLARIFICATION_MODELS}


def _get_ssl_contexts() -> List[ssl.SSLContext]:
    """
    Build SSL contexts for model downloads:
    1. First try Mozilla CA bundle via `certifi` (or OS default store).
    2. Fallback to `_create_unverified_context()` so macOS PyInstaller builds
       (where `/Library/Frameworks/Python.framework/.../cert.pem` is absent)
       and machines behind corporate TLS inspection proxies never fail with
       `CERTIFICATE_VERIFY_FAILED`.
    """
    contexts: List[ssl.SSLContext] = []
    if HAS_CERTIFI:
        try:
            contexts.append(ssl.create_default_context(cafile=certifi.where()))
        except Exception:
            pass
    try:
        contexts.append(ssl.create_default_context())
    except Exception:
        pass
    try:
        contexts.append(ssl._create_unverified_context())
    except Exception:
        pass
    return contexts


def _download_url_to_path(url: str, dest: Path, label: str = "") -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    last_err = None
    short_name = label or dest.name

    for ctx in _get_ssl_contexts():
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "WispAI/1.0"})
            with urllib.request.urlopen(req, timeout=300, context=ctx) as resp, open(
                tmp_path, "wb"
            ) as out_f:
                total = int(resp.headers.get("Content-Length", "0") or "0")
                downloaded = 0
                last_emit = 0.0
                while True:
                    chunk = resp.read(1024 * 512)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if _PROGRESS_HOOK and (now - last_emit >= 0.35 or (total and downloaded >= total)):
                        last_emit = now
                        dl_mb = downloaded / (1024 * 1024)
                        if total > 0:
                            tot_mb = total / (1024 * 1024)
                            pct = min(100, int(round((downloaded / total) * 100)))
                            _PROGRESS_HOOK(
                                f"Downloading {short_name}: {dl_mb:.1f} / {tot_mb:.1f} MB ({pct}%)"
                            )
                        else:
                            _PROGRESS_HOOK(f"Downloading {short_name}: {dl_mb:.1f} MB...")
            tmp_path.replace(dest)
            return dest
        except (urllib.error.URLError, ssl.SSLError) as err:
            last_err = err
            if "CERTIFICATE_VERIFY_FAILED" in str(err) or isinstance(err, ssl.SSLError):
                continue
            raise

    if last_err is not None:
        raise last_err
    raise RuntimeError(f"Failed to download {url}")


def hf_hub_download(repo_id: str, filename: str, local_dir: str) -> str:
    """Download a public model file directly from Hugging Face via standard-library urllib (zero native DLL dependencies)."""
    dest = Path(local_dir) / filename
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    return str(_download_url_to_path(url, dest, label=f"{repo_id.split('/')[-1]}/{Path(filename).name}"))


def get_whisper_model_dir(model_id: str) -> Path:
    if model_id == "large-v3-turbo":
        model_id = "turbo"
    elif model_id == "hviske-v2-fp16":
        model_id = "hviske-v2-int8"
    return config.MODELS_DIR / f"whisper-onnx-{model_id}"


def get_whisper_onnx_paths(model_id: str) -> Dict[str, Path]:
    if model_id == "large-v3-turbo":
        model_id = "turbo"
    elif model_id == "hviske-v2-fp16":
        model_id = "hviske-v2-int8"
    info = WHISPER_MAP.get(model_id, WHISPER_MAP[DEFAULT_WHISPER_MODEL])
    d = get_whisper_model_dir(model_id)
    if info.get("engine") == "hf-whisper-onnx":
        enc_rel = info.get("encoder_file", "onnx/encoder_model_q4.onnx")
        dec_rel = info.get("decoder_file", "onnx/decoder_model_merged_q4.onnx")
        return {
            "dir": d,
            "encoder": d / enc_rel,
            "decoder": d / dec_rel,
            "tokens": d / "tokenizer.json",
            "engine": Path("hf-whisper-onnx"),
        }
    if info.get("engine") == "wav2vec2-onnx":
        onnx_rel = info.get("onnx_file", "onnx/model_quantized.onnx")
        return {
            "dir": d,
            "encoder": d / onnx_rel,
            "decoder": d / onnx_rel,
            "tokens": d / "vocab.json",
            "engine": Path("wav2vec2-onnx"),
        }
    prefix = info["prefix"]
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


def get_clarification_model_dir(model_id: str) -> Path:
    return config.MODELS_DIR / f"clarifier-{model_id}"


def is_clarification_model_downloaded(model_id: str) -> bool:
    if model_id in ("none", "whisper-context"):
        return True
    info = CLARIFICATION_MAP.get(model_id)
    if not info or info.get("type") != "qwen-onnx":
        return True
    d = get_clarification_model_dir(model_id)
    onnx_rel = info.get("onnx_file", "onnx/model_q4.onnx")
    return (
        (d / onnx_rel).exists()
        and (d / "tokenizer.json").exists()
        and (d / "config.json").exists()
    )


def ensure_clarification_model(model_id: str = DEFAULT_CLARIFICATION_MODEL) -> Path:
    """Download a local ONNX Context Clarification LLM (e.g. Qwen2.5-0.5B or Qwen2.5-1.5B ONNX Q4)."""
    if model_id in ("none", "whisper-context"):
        return config.MODELS_DIR
    info = CLARIFICATION_MAP.get(model_id)
    if not info or info.get("type") != "qwen-onnx":
        return config.MODELS_DIR
    target_dir = get_clarification_model_dir(model_id)
    if is_clarification_model_downloaded(model_id):
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    repo_id = info["hf_repo"]
    onnx_rel = info.get("onnx_file", "onnx/model_q4.onnx")
    print(f"[ModelManager] Downloading Context Clarifier '{model_id}' from {repo_id} to {target_dir} ...")
    for fname in ("config.json", "tokenizer.json", "tokenizer_config.json", onnx_rel):
        hf_hub_download(repo_id=repo_id, filename=fname, local_dir=str(target_dir))
    return target_dir


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
    target_dir = paths["dir"]
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ModelManager] Downloading Whisper ONNX '{model_id}' from {repo_id} to {target_dir} ...")
    if info.get("engine") == "hf-whisper-onnx":
        enc_rel = info.get("encoder_file", "onnx/encoder_model_q4.onnx")
        dec_rel = info.get("decoder_file", "onnx/decoder_model_merged_q4.onnx")
        files_to_fetch = [
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            enc_rel,
            dec_rel,
        ]
    elif info.get("engine") == "wav2vec2-onnx":
        onnx_rel = info.get("onnx_file", "onnx/model_quantized.onnx")
        files_to_fetch = [
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "added_tokens.json",
            "special_tokens_map.json",
            onnx_rel,
        ]
    else:
        prefix = info["prefix"]
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
    return _download_url_to_path(
        SPEAKER_EMBEDDING_MODEL_URL, config.SPEAKER_EMBEDDING_MODEL_PATH
    )


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
    elif model_type == "clarification":
        if model_id and model_id not in ("none", "whisper-context"):
            d = get_clarification_model_dir(model_id)
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
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

    clarification_list: List[Dict] = []
    for cm in CLARIFICATION_MODELS:
        cid = cm["id"]
        dl = is_clarification_model_downloaded(cid)
        if cid == "none":
            lpath = "Skipped (Raw Speech-to-Text)"
        elif cid == "whisper-context":
            lpath = "Built into Whisper Context Validator"
        else:
            lpath = str(get_clarification_model_dir(cid))
        clarification_list.append({**cm, "downloaded": dl, "local_path": lpath})

    translation_list: List[Dict] = []
    for tm in TRANSLATION_MODELS:
        if tm["id"] == "opus-mt-da-en":
            dl = is_opus_da_en_ready()
            lpath = str(get_opus_pair_dir("da-en"))
        elif tm["id"] == "nllb-200-600M":
            dl = is_nllb_model_downloaded()
            lpath = str(config.NLLB_LOCAL_DIR)
        elif tm["id"] == "none":
            dl = True
            lpath = "Skipped (Transcription Only)"
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
        "needs_onboarding": not (spk_ready and any_whisper_ready),
        "whisper_models": whisper_list,
        "clarification_models": clarification_list,
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
