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
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_WHISPER_MODEL,
    NLLB_REPO_ID,
    OPUS_PAIR_REPOS,
    SPEAKER_DIARIZATION_MODELS,
    SPEAKER_EMBEDDING_MODEL_URL,
    TRANSLATION_MODELS,
    WHISPER_MODELS,
)

WHISPER_MAP = {m["id"]: m for m in WHISPER_MODELS}
CLARIFICATION_MAP = {m["id"]: m for m in CLARIFICATION_MODELS}
DIARIZATION_MAP = {m["id"]: m for m in SPEAKER_DIARIZATION_MODELS}


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


def _safe_custom_model_dir(folder_name: str) -> Optional[Path]:
    """Safely resolve a direct child folder inside `config.MODELS_DIR` (blocks path traversal)."""
    clean = (folder_name or "").strip()
    if not clean or "/" in clean or "\\" in clean or ".." in clean or clean in (".", "hf_cache"):
        return None
    root = config.MODELS_DIR.resolve()
    cand = (root / clean).resolve()
    if cand.parent != root or not cand.is_dir():
        return None
    return cand


def _detect_custom_stt_paths(folder: Path) -> Optional[Dict[str, Path]]:
    """
    Auto-detect whether an arbitrary local folder inside `models/` is a usable
    Speech-to-Text ONNX model (`hf-whisper-onnx`, `wav2vec2-onnx`, or `sherpa-onnx`).
    """
    if not folder.is_dir():
        return None

    onnx_dir = folder / "onnx"
    # 1. HuggingFace Optimum Whisper ONNX (encoder_model*.onnx + decoder_model_merged*.onnx + tokenizer.json)
    if (folder / "tokenizer.json").exists() and onnx_dir.is_dir():
        encoders = sorted(onnx_dir.glob("encoder_model*.onnx"))
        decoders = sorted(onnx_dir.glob("decoder_model_merged*.onnx")) or sorted(
            onnx_dir.glob("decoder_model*.onnx")
        )
        if encoders and decoders:
            return {
                "dir": folder,
                "encoder": encoders[0],
                "decoder": decoders[0],
                "tokens": folder / "tokenizer.json",
                "engine": Path("hf-whisper-onnx"),
            }

    # 2. Wav2Vec2 CTC ONNX (vocab.json + onnx/model*.onnx or *.onnx)
    if (folder / "vocab.json").exists():
        w2v_files = sorted(onnx_dir.glob("*.onnx")) if onnx_dir.is_dir() else []
        if not w2v_files:
            w2v_files = sorted(folder.glob("*.onnx"))
        if w2v_files:
            return {
                "dir": folder,
                "encoder": w2v_files[0],
                "decoder": w2v_files[0],
                "tokens": folder / "vocab.json",
                "engine": Path("wav2vec2-onnx"),
            }

    # 3. Sherpa-ONNX Whisper (*-encoder*.onnx + *-decoder*.onnx + *-tokens.txt)
    s_enc = sorted(folder.glob("*-encoder*.onnx"))
    s_dec = sorted(folder.glob("*-decoder*.onnx"))
    s_tok = sorted(folder.glob("*-tokens.txt"))
    if s_enc and s_dec and s_tok:
        return {
            "dir": folder,
            "encoder": s_enc[0],
            "decoder": s_dec[0],
            "tokens": s_tok[0],
        }

    return None


def _dir_size_mb(folder: Path) -> int:
    total = 0
    try:
        for p in folder.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return max(1, int(round(total / (1024 * 1024)))) if total > 0 else 0


def _discover_unmanaged_local_models() -> List[Dict]:
    """
    Scan `config.MODELS_DIR` for any folders not in the built-in catalog:
    - Custom user-added ONNX Speech-to-Text models (automatically usable!)
    - Old / legacy / removed model folders left on disk (easy 1-click cleanup!)
    """
    if not config.MODELS_DIR.exists():
        return []

    known_dirs = {"hf_cache", "speaker_embedding", config.NLLB_LOCAL_DIR.name}
    for m in WHISPER_MODELS:
        known_dirs.add(get_whisper_model_dir(m["id"]).name)
    for cm in CLARIFICATION_MODELS:
        if cm["id"] not in ("none", "whisper-context"):
            known_dirs.add(get_clarification_model_dir(cm["id"]).name)
    for pair in OPUS_PAIR_REPOS.keys():
        known_dirs.add(get_opus_pair_dir(pair).name)

    found: List[Dict] = []
    for entry in sorted(config.MODELS_DIR.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_dir() or entry.name in known_dirs or entry.name.startswith("."):
            continue
        stt_paths = _detect_custom_stt_paths(entry)
        size_mb = _dir_size_mb(entry)
        if stt_paths is not None:
            engine = str(stt_paths.get("engine", "sherpa-onnx"))
            custom_id = f"custom:{entry.name}"
            found.append(
                {
                    "folder_name": entry.name,
                    "custom_whisper_id": custom_id,
                    "kind": "custom_stt",
                    "engine": engine,
                    "size_mb": str(size_mb),
                    "local_path": str(entry),
                }
            )
        else:
            found.append(
                {
                    "folder_name": entry.name,
                    "custom_whisper_id": "",
                    "kind": "legacy",
                    "engine": "unknown",
                    "size_mb": str(size_mb),
                    "local_path": str(entry),
                }
            )
    return found


def get_whisper_model_dir(model_id: str) -> Path:
    if model_id.startswith("custom:"):
        custom_dir = _safe_custom_model_dir(model_id[len("custom:") :])
        if custom_dir is not None:
            return custom_dir
        return config.MODELS_DIR / model_id[len("custom:") :]
    if model_id == "large-v3-turbo":
        model_id = "turbo"
    elif model_id == "hviske-v2-fp16":
        model_id = "hviske-v2-int8"
    return config.MODELS_DIR / f"whisper-onnx-{model_id}"


def get_whisper_onnx_paths(model_id: str) -> Dict[str, Path]:
    if model_id.startswith("custom:"):
        custom_dir = _safe_custom_model_dir(model_id[len("custom:") :])
        if custom_dir is not None:
            detected = _detect_custom_stt_paths(custom_dir)
            if detected is not None:
                return detected
        missing = config.MODELS_DIR / "__missing__"
        return {"dir": missing, "encoder": missing, "decoder": missing, "tokens": missing}
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


def get_diarization_model_path(model_id: Optional[str] = None) -> Optional[Path]:
    mid = (model_id or config.get_active_diarization_model()).strip()
    if mid == "none":
        return None
    info = DIARIZATION_MAP.get(mid) or DIARIZATION_MAP.get(DEFAULT_DIARIZATION_MODEL)
    if not info or not info.get("filename"):
        return None
    return config.MODELS_DIR / "speaker_embedding" / info["filename"]


def is_diarization_model_downloaded(model_id: Optional[str] = None) -> bool:
    mid = (model_id or config.get_active_diarization_model()).strip()
    if mid == "none":
        return True
    p = get_diarization_model_path(mid)
    return bool(p is not None and p.exists() and p.stat().st_size > 1_000_000)


def is_speaker_model_downloaded(model_id: Optional[str] = None) -> bool:
    if model_id is not None:
        return is_diarization_model_downloaded(model_id)
    if is_diarization_model_downloaded(config.get_active_diarization_model()):
        return True
    return any(
        is_diarization_model_downloaded(m["id"])
        for m in SPEAKER_DIARIZATION_MODELS
        if m["id"] != "none"
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
    extra_rels = [f.strip() for f in info.get("onnx_extra_files", "").split(",") if f.strip()]
    return (
        (d / onnx_rel).exists()
        and all((d / ef).exists() for ef in extra_rels)
        and (d / "tokenizer.json").exists()
        and (d / "config.json").exists()
    )


def ensure_clarification_model(model_id: str = DEFAULT_CLARIFICATION_MODEL) -> Path:
    """Download a local ONNX Context Clarification LLM (e.g. DanskGPT, EuroLLM, Gemma 3, Qwen 3, Llama 3.2, Phi-4)."""
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
    extra_rels = [f.strip() for f in info.get("onnx_extra_files", "").split(",") if f.strip()]
    print(f"[ModelManager] Downloading Context Clarifier '{model_id}' from {repo_id} to {target_dir} ...")
    for fname in ("config.json", "tokenizer.json", "tokenizer_config.json", onnx_rel, *extra_rels):
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
    elif model_id == "hviske-v2-fp16":
        model_id = "hviske-v2-int8"
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


def ensure_speaker_model(model_id: Optional[str] = None) -> Path:
    """Ensure the requested ONNX speaker embedding model is stored locally."""
    mid = (model_id or config.get_active_diarization_model()).strip()
    if mid == "none":
        return config.MODELS_DIR / "speaker_embedding"
    info = DIARIZATION_MAP.get(mid) or DIARIZATION_MAP.get(DEFAULT_DIARIZATION_MODEL)
    assert info is not None
    target_path = config.MODELS_DIR / "speaker_embedding" / info["filename"]
    if target_path.exists() and target_path.stat().st_size > 1_000_000:
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[ModelManager] Downloading Speaker Embedding ONNX '{mid}' to {target_path} ..."
    )
    return _download_url_to_path(info["url"], target_path)


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
    """Delete a downloaded local AI model (or custom/legacy model folder) from disk to free up storage."""
    if model_type == "custom":
        target_dir = _safe_custom_model_dir(model_id)
        if target_dir is not None and target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
    elif model_type == "whisper":
        if model_id.startswith("custom:"):
            target_dir = _safe_custom_model_dir(model_id[len("custom:") :])
        else:
            target_dir = get_whisper_model_dir(model_id)
        if target_dir is not None and target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
    elif model_type in ("opus", "opus_pair", "opus-mt-da-en") or (
        model_type == "translation" and model_id == "opus-mt-da-en"
    ):
        if model_id and model_id in OPUS_PAIR_REPOS:
            d = get_opus_pair_dir(model_id)
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        else:
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
        if model_id and model_id in DIARIZATION_MAP:
            p = get_diarization_model_path(model_id)
            if p is not None and p.exists():
                p.unlink(missing_ok=True)
        elif config.SPEAKER_EMBEDDING_MODEL_PATH.exists():
            config.SPEAKER_EMBEDDING_MODEL_PATH.unlink(missing_ok=True)


def get_system_hardware_recommendations() -> Dict:
    """Detect local CPU threads and RAM via standard library and return tailored model recommendations."""
    import ctypes
    import os
    import platform
    import sys

    cpu_threads = os.cpu_count() or 4
    cpu_name = platform.processor() or "x64 Processor"
    total_ram_gb = 8.0
    avail_ram_gb = 4.0
    gpu_name = ""

    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\Description\System\CentralProcessor\0",
            )
            val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            if val and str(val).strip():
                cpu_name = str(val).strip()
        except Exception:
            pass

        # Detect primary hardware GPU via Windows Display Adapter registry class (zero subprocess overhead)
        try:
            import winreg

            disp_class = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
            candidates: List[str] = []
            for sub_i in range(8):
                try:
                    k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, f"{disp_class}\\{sub_i:04d}")
                    desc, _ = winreg.QueryValueEx(k, "DriverDesc")
                    desc_s = str(desc).strip()
                    if desc_s and "microsoft basic" not in desc_s.lower() and "virtual" not in desc_s.lower():
                        candidates.append(desc_s)
                except Exception:
                    continue
            if candidates:
                # Prefer discrete NVIDIA / AMD Radeon / Intel Arc over integrated graphics
                discrete = [
                    c for c in candidates
                    if any(b in c.lower() for b in ("nvidia", "geforce", "rtx", "gtx", "quadro", "radeon", "arc"))
                ]
                gpu_name = discrete[0] if discrete else candidates[0]
        except Exception:
            pass

        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total_ram_gb = round(stat.ullTotalPhys / (1024**3), 1)
                avail_ram_gb = round(stat.ullAvailPhys / (1024**3), 1)
        except Exception:
            pass
    elif sys.platform == "darwin":
        try:
            import subprocess

            brand = (
                subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    stderr=subprocess.DEVNULL,
                )
                .decode("utf-8", errors="ignore")
                .strip()
            )
            if brand:
                cpu_name = brand
        except Exception:
            pass
        try:
            import subprocess

            mem_bytes = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"],
                    stderr=subprocess.DEVNULL,
                )
                .decode("utf-8", errors="ignore")
                .strip()
            )
            if mem_bytes > 0:
                total_ram_gb = round(mem_bytes / (1024**3), 1)
                avail_ram_gb = round(total_ram_gb * 0.65, 1)
        except Exception:
            pass
    else:
        try:
            if Path("/proc/cpuinfo").exists():
                for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
                    if "model name" in line and ":" in line:
                        cname = line.split(":", 1)[1].strip()
                        if cname:
                            cpu_name = cname
                            break
        except Exception:
            pass
        try:
            if Path("/proc/meminfo").exists():
                for line in Path("/proc/meminfo").read_text().splitlines():
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        total_ram_gb = round(kb / (1024**2), 1)
                    elif line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        avail_ram_gb = round(kb / (1024**2), 1)
        except Exception:
            pass

    gpu_provider = config.get_preferred_gpu_provider()
    gpu_detected = bool(gpu_name) or bool(gpu_provider)
    gpu_enabled = config.get_gpu_acceleration_enabled() if gpu_detected else False
    has_discrete_gpu = gpu_enabled and any(
        b in gpu_name.lower()
        for b in ("nvidia", "geforce", "rtx", "gtx", "quadro", "radeon", "rx ", "arc")
    )

    if (has_discrete_gpu and total_ram_gb >= 7.5) or (
        gpu_enabled and total_ram_gb >= 15.0
    ):
        tier = "high"
        tier_label = "GPU-Accelerated Workstation"
        rec_multilingual = "turbo"
        rec_danish = "hviske-v2-int8"
        rec_clarifier = "qwen2.5-1.5b" if total_ram_gb >= 15.0 else "qwen2.5-0.5b"
        recommended_ids = [
            "turbo",
            "large-v3",
            "hviske-v2-int8",
            "roest-v3-315m",
            "distil-large-v3.5",
            "kb-whisper-medium",
            "whisper-turbo-de",
            "whisper-es",
            "whisper-medium-fr",
        ]
        summary = (
            f"GPU Acceleration is enabled ({gpu_name or gpu_provider}). Your system can comfortably run "
            "the largest high-precision models (Large-v3-Turbo, Hviske v2 INT8, and Qwen Context Clarifier)."
        )
    elif total_ram_gb >= 15.0 and cpu_threads >= 8:
        tier = "high"
        tier_label = "High-Performance CPU Workstation"
        rec_multilingual = "turbo"
        rec_danish = "hviske-v2-int8"
        rec_clarifier = "whisper-context"
        recommended_ids = [
            "turbo",
            "hviske-v2-int8",
            "roest-v3-315m",
            "distil-large-v3.5",
            "kb-whisper-medium",
            "whisper-turbo-de",
        ]
        summary = (
            f"Your CPU has strong specs ({cpu_threads} threads, {total_ram_gb} GB RAM) and can run "
            "high-precision models like Hviske v2 INT8, Røst v3 315M, and Whisper Large-v3-Turbo."
        )
    elif (gpu_enabled and total_ram_gb >= 6.0) or (
        total_ram_gb >= 7.5 and cpu_threads >= 6
    ):
        tier = "balanced"
        tier_label = (
            "GPU-Assisted Balanced PC" if gpu_enabled else "Balanced Desktop / Laptop"
        )
        rec_multilingual = "small"
        rec_danish = "hviske-v2"
        rec_clarifier = "whisper-context"
        recommended_ids = [
            "small",
            "hviske-v2",
            "roest-v3-315m",
            "nb-whisper-small",
            "whisper-small-it",
        ]
        summary = (
            f"Your PC ({cpu_threads} threads, {total_ram_gb} GB RAM) is well-suited for 'Hviske v2 (Q4)' or "
            "'Røst v3 315M' for Danish, and 'Whisper Small' for fast multi-language transcription."
        )
    else:
        tier = "light"
        tier_label = "Lightweight Laptop / Low-Memory PC"
        rec_multilingual = "base"
        rec_danish = "roest-315m"
        rec_clarifier = "whisper-context"
        recommended_ids = ["base", "small", "roest-315m", "nb-whisper-small", "whisper-small-it"]
        summary = (
            f"For smooth performance on {cpu_threads} threads and {total_ram_gb} GB RAM, we recommend "
            "'Røst 315M (~303 MB)' for fast Danish transcription and 'Whisper Base / Small' for multi-language."
        )

    return {
        "cpu_name": cpu_name,
        "cpu_threads": cpu_threads,
        "total_ram_gb": total_ram_gb,
        "avail_ram_gb": avail_ram_gb,
        "gpu_name": gpu_name,
        "gpu_detected": gpu_detected,
        "gpu_provider": gpu_provider,
        "gpu_acceleration_enabled": gpu_enabled,
        "tier": tier,
        "tier_label": tier_label,
        "recommended_multilingual_id": rec_multilingual,
        "recommended_danish_id": rec_danish,
        "recommended_clarifier_id": rec_clarifier,
        "recommended_whisper_ids": recommended_ids,
        "summary": summary,
    }


def get_models_status() -> Dict:
    """Return readiness status of all local AI models for the UI."""
    # Auto-clean legacy broken whisper-onnx-hviske-v2-fp16 folder if present on disk/network share
    legacy_fp16_dir = config.MODELS_DIR / "whisper-onnx-hviske-v2-fp16"
    if legacy_fp16_dir.exists():
        try:
            shutil.rmtree(legacy_fp16_dir, ignore_errors=True)
        except Exception:
            pass

    whisper_list: List[Dict] = []
    for m in WHISPER_MODELS:
        whisper_list.append(
            {
                **m,
                "downloaded": is_whisper_model_downloaded(m["id"]),
                "local_path": str(get_whisper_model_dir(m["id"])),
            }
        )

    # Auto-discover custom user-added ONNX models & old/legacy model folders in `models/`
    unmanaged_models = _discover_unmanaged_local_models()
    for um in unmanaged_models:
        if um["kind"] == "custom_stt" and um["custom_whisper_id"]:
            whisper_list.append(
                {
                    "id": um["custom_whisper_id"],
                    "name": f"{um['folder_name']} (Custom Local ONNX, ~{um['size_mb']} MB)",
                    "description": f"User-supplied local {um['engine']} Speech-to-Text model discovered in your models folder.",
                    "size_mb": um["size_mb"],
                    "language_group": "custom",
                    "downloaded": True,
                    "is_custom": True,
                    "local_path": um["local_path"],
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

    opus_pairs_status: Dict[str, Dict] = {}
    for pair_key, repo_id in OPUS_PAIR_REPOS.items():
        opus_pairs_status[pair_key] = {
            "pair": pair_key,
            "repo": repo_id,
            "size_mb": "85",
            "downloaded": is_opus_pair_downloaded(pair_key),
            "local_path": str(get_opus_pair_dir(pair_key)),
        }
    any_opus_ready = any(v["downloaded"] for v in opus_pairs_status.values())

    translation_list: List[Dict] = []
    for tm in TRANSLATION_MODELS:
        if tm["id"] == "opus-mt-da-en":
            dl = any_opus_ready
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

    active_diar_id = config.get_active_diarization_model()
    speaker_diarization_models: List[Dict] = []
    for sm in SPEAKER_DIARIZATION_MODELS:
        sid = sm["id"]
        dl = is_diarization_model_downloaded(sid)
        p = get_diarization_model_path(sid)
        speaker_diarization_models.append(
            {
                **sm,
                "downloaded": dl,
                "local_path": str(p) if p is not None else "Single-Speaker Mode (Disabled)",
            }
        )

    spk_ready = is_speaker_model_downloaded()
    any_whisper_ready = any(w["downloaded"] for w in whisper_list)
    any_neural_translator_ready = any_opus_ready or is_nllb_model_downloaded()
    active_spk_info = DIARIZATION_MAP.get(active_diar_id) or SPEAKER_DIARIZATION_MODELS[0]
    active_spk_path = get_diarization_model_path(active_diar_id)

    return {
        "storage_directory": str(config.DATA_DIR),
        "models_directory": str(config.MODELS_DIR),
        "sessions_directory": str(config.SESSIONS_DIR),
        "needs_onboarding": not (spk_ready and any_whisper_ready),
        "system_specs": get_system_hardware_recommendations(),
        "whisper_models": whisper_list,
        "clarification_models": clarification_list,
        "translation_models": translation_list,
        "opus_pairs_status": opus_pairs_status,
        "local_unmanaged_models": unmanaged_models,
        "active_diarization_model": active_diar_id,
        "speaker_diarization_models": speaker_diarization_models,
        "speaker_diarization": {
            "id": active_spk_info["id"],
            "name": active_spk_info["name"],
            "size_mb": active_spk_info["size_mb"],
            "downloaded": is_diarization_model_downloaded(active_diar_id),
            "local_path": str(active_spk_path) if active_spk_path else "Single-Speaker Mode",
        },
        "translator": {
            "id": "opus-mt-da-en",
            "name": "OPUS-MT Language Pair Specialists + NLLB-200",
            "size_mb": "85",
            "downloaded": any_neural_translator_ready,
            "local_path": str(get_opus_pair_dir("da-en")),
        },
    }
