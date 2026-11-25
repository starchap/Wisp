"""
Configuration, paths, and supported languages for Wishper Local AI Engine.
All AI models and session recordings are stored locally in the `data/` directory.
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

if getattr(sys, "frozen", False) or os.environ.get("WISP_PACKAGED") == "1":
    BASE_DIR = Path.home() / ".wisp"
    BASE_DIR.mkdir(parents=True, exist_ok=True)
else:
    BASE_DIR = Path(__file__).resolve().parent.parent

SETTINGS_FILE = BASE_DIR / "user_settings.json"


def _load_user_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_user_settings(patch: dict) -> dict:
    data = _load_user_settings()
    data.update(patch)
    SETTINGS_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return data


def _load_initial_data_dir() -> Path:
    custom = str(_load_user_settings().get("storage_path", "")).strip()
    if custom:
        try:
            p = Path(custom).resolve()
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            pass
    return BASE_DIR / "data"


def get_gpu_acceleration_enabled() -> bool:
    """Return whether GPU acceleration is enabled (defaults to True when a GPU is present)."""
    val = _load_user_settings().get("gpu_acceleration")
    if isinstance(val, bool):
        return val
    return True


def set_gpu_acceleration_enabled(enabled: bool) -> bool:
    """Persist GPU acceleration toggle in `user_settings.json`."""
    _save_user_settings({"gpu_acceleration": bool(enabled)})
    return bool(enabled)


def get_active_diarization_model() -> str:
    """Read the user's selected SpeechSplitter / Diarization model ID from `user_settings.json`."""
    try:
        if SETTINGS_FILE.exists():
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            val = str(data.get("diarization_model", "")).strip()
            if val:
                return val
    except Exception:
        pass
    # Backwards-compatibility: if legacy 3D-Speaker ERes2Net is already downloaded on disk, keep it active
    try:
        legacy_path = MODELS_DIR / "speaker_embedding" / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
        camp_path = MODELS_DIR / "speaker_embedding" / "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
        if legacy_path.exists() and not camp_path.exists():
            return "3dspeaker-eres2net"
    except Exception:
        pass
    return DEFAULT_DIARIZATION_MODEL


def set_active_diarization_model(model_id: str) -> str:
    """Persist the user's active SpeechSplitter / Diarization model ID in `user_settings.json`."""
    clean_id = (model_id or DEFAULT_DIARIZATION_MODEL).strip()
    _save_user_settings({"diarization_model": clean_id})
    return clean_id


def get_preferred_gpu_provider() -> str:
    """Detect the highest-priority ONNX Runtime GPU execution provider available on this machine."""
    try:
        import onnxruntime as ort

        avail = set(ort.get_available_providers())
        for prov in ("CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider"):
            if prov in avail:
                return prov
    except Exception:
        pass
    return ""


def create_ort_session(
    model_path: str,
    sess_opts=None,
    is_autoregressive_decoder: bool = False,
):
    """
    Create an `onnxruntime.InferenceSession` using GPU acceleration when enabled & available
    (CUDA / DirectML / CoreML), with guaranteed safe automatic fallback to CPUExecutionProvider
    if any model op or driver issue occurs on GPU.

    DirectML-specific optimizations:
    - Autoregressive token-by-token decoders with dynamic KV-cache shapes run ~4-5x faster on
      CPU than DirectML (which suffers per-token PCIe copy & graph recompilation overhead),
      while the heavy parallel encoder runs ~15-17x faster on DirectML GPU. Using Hybrid
      GPU Encoder + CPU Decoder delivers the fastest end-to-end transcription on Windows.
    - Sherpa-ONNX legacy dynamic-INT8 graphs (`*.int8.onnx`, dot before int8) can trigger a
      native C++ abort in DirectML, so they are routed safely to CPUExecutionProvider.
      (Optimum HuggingFace ONNX models `*_int8.onnx`, `*_q4.onnx`, `*_fp16.onnx` run cleanly on DirectML.)
    """
    import onnxruntime as ort

    if sess_opts is None:
        sess_opts = ort.SessionOptions()

    gpu_prov = get_preferred_gpu_provider() if get_gpu_acceleration_enabled() else ""
    if gpu_prov == "DmlExecutionProvider":
        fname = Path(model_path).name.lower()
        if is_autoregressive_decoder:
            gpu_prov = ""
        elif fname.endswith(".int8.onnx"):
            print(
                f"[ONNX] Routing Sherpa-ONNX dynamic-INT8 model '{Path(model_path).name}' "
                f"to CPU (incompatible with DirectML)."
            )
            gpu_prov = ""

    if gpu_prov:
        try:
            gpu_opts = ort.SessionOptions()
            gpu_opts.intra_op_num_threads = sess_opts.intra_op_num_threads
            gpu_opts.graph_optimization_level = sess_opts.graph_optimization_level
            if gpu_prov == "DmlExecutionProvider":
                # DirectML best practices for stability and memory management
                gpu_opts.enable_mem_pattern = False
                gpu_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            return ort.InferenceSession(
                str(model_path),
                sess_options=gpu_opts,
                providers=[gpu_prov, "CPUExecutionProvider"],
            )
        except Exception as gpu_err:
            print(
                f"[ONNX] GPU provider '{gpu_prov}' failed for '{Path(model_path).name}' "
                f"({gpu_err}); falling back safely to CPU."
            )

    return ort.InferenceSession(
        str(model_path),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )


DATA_DIR = _load_initial_data_dir()
MODELS_DIR = DATA_DIR / "models"
SESSIONS_DIR = DATA_DIR / "sessions"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Set HuggingFace cache to our local models directory so everything stays self-contained
# and strictly disable all external telemetry/tracking
os.environ["HF_HOME"] = str(MODELS_DIR / "hf_cache")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["DISABLE_TELEMETRY"] = "1"


def set_storage_path(new_path_str: str) -> Path:
    """Update the root data storage path (or models parent path) and persist it."""
    global DATA_DIR, MODELS_DIR, SESSIONS_DIR, NLLB_LOCAL_DIR, SPEAKER_EMBEDDING_MODEL_PATH
    raw = Path(new_path_str.strip()).resolve()
    # If the user selected a folder literally named "models", use its parent as DATA_DIR
    if raw.name.lower() == "models":
        DATA_DIR = raw.parent
    else:
        DATA_DIR = raw

    MODELS_DIR = DATA_DIR / "models"
    SESSIONS_DIR = DATA_DIR / "sessions"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    NLLB_LOCAL_DIR = MODELS_DIR / "nllb-200-onnx-int8"
    SPEAKER_EMBEDDING_MODEL_PATH = (
        MODELS_DIR
        / "speaker_embedding"
        / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
    )
    os.environ["HF_HOME"] = str(MODELS_DIR / "hf_cache")

    _save_user_settings({"storage_path": str(DATA_DIR)})
    return DATA_DIR

# Supported Whisper sizes (Sherpa-ONNX INT8)
WHISPER_MODELS: List[Dict[str, str]] = [
    {
        "id": "tiny",
        "name": "Whisper Tiny (~110 MB) — English / Quick Test",
        "description": "Fastest, lowest memory usage. Best for English; use Small or Turbo for Danish.",
        "size_mb": "110",
        "hf_repo": "csukuangfj/sherpa-onnx-whisper-tiny",
        "prefix": "tiny",
        "language_group": "multilingual",
    },
    {
        "id": "base",
        "name": "Whisper Base (~195 MB)",
        "description": "Fast & lightweight for clear speech.",
        "size_mb": "195",
        "hf_repo": "csukuangfj/sherpa-onnx-whisper-base",
        "prefix": "base",
        "language_group": "multilingual",
    },
    {
        "id": "small",
        "name": "Whisper Small (~610 MB) — Recommended for Danish",
        "description": "Great balance of live speed and strong Danish, Spanish, German & English accuracy.",
        "size_mb": "610",
        "hf_repo": "csukuangfj/sherpa-onnx-whisper-small",
        "prefix": "small",
        "language_group": "multilingual",
    },
    {
        "id": "medium",
        "name": "Whisper Medium (~1.7 GB) — High Accuracy Danish",
        "description": "High accuracy for Danish dialects, strong accents, and noisy meetings.",
        "size_mb": "1700",
        "hf_repo": "csukuangfj/sherpa-onnx-whisper-medium",
        "prefix": "medium",
        "language_group": "multilingual",
    },
    {
        "id": "turbo",
        "name": "Whisper Large-v3-Turbo (~1.6 GB) — Fast Large Multilingual",
        "description": "Tip-top Large-v3 Danish & multilingual accuracy at fast real-time speed.",
        "size_mb": "1600",
        "hf_repo": "csukuangfj/sherpa-onnx-whisper-turbo",
        "prefix": "turbo",
        "language_group": "multilingual",
    },
    {
        "id": "large-v3",
        "name": "Whisper Large-v3 Full 32-Layer (~1.77 GB) — Deep Multilingual & Danish",
        "description": "Full 32-decoder-layer OpenAI Whisper Large-v3 (INT8) for maximum context depth in overlapping speech.",
        "size_mb": "1775",
        "hf_repo": "csukuangfj/sherpa-onnx-whisper-large-v3",
        "prefix": "large-v3",
        "language_group": "multilingual",
    },
    {
        "id": "hviske-v2",
        "name": "Hviske v2 — Danish Specialist (CoRal Whisper Large-v3 Q4, ~1.06 GB)",
        "description": "Specialized Danish Speech-to-Text model (Syv.ai / CoRal) fine-tuned specifically on Danish dialects, compound words, and landmarks.",
        "size_mb": "1065",
        "hf_repo": "varsan-g/hviske-v2-onnx",
        "prefix": "hviske-v2",
        "engine": "hf-whisper-onnx",
        "encoder_file": "onnx/encoder_model_q4.onnx",
        "decoder_file": "onnx/decoder_model_merged_q4.onnx",
        "language_group": "da",
    },
    {
        "id": "hviske-v2-int8",
        "name": "Hviske v2 High-Precision — Danish Specialist (CoRal Whisper Large-v3 INT8, ~1.35 GB)",
        "description": "Higher-precision 8-bit decoder Hviske v2 Danish Specialist for sharper Danish word endings, names, and complex interviews.",
        "size_mb": "1345",
        "hf_repo": "varsan-g/hviske-v2-onnx",
        "prefix": "hviske-v2-int8",
        "engine": "hf-whisper-onnx",
        "encoder_file": "onnx/encoder_model_q4.onnx",
        "decoder_file": "onnx/decoder_model_merged_int8.onnx",
        "language_group": "da",
    },
    {
        "id": "roest-315m",
        "name": "Røst 315M — Danish Fast Acoustic Specialist (CoRal Wav2Vec2 INT8, ~303 MB)",
        "description": "Ultra-fast native Danish acoustic speech recognizer (Alexandra Instituttet / CoRal project) trained purely on Danish speech.",
        "size_mb": "303",
        "hf_repo": "PierreMesure/roest-315m-onnx",
        "prefix": "roest-315m",
        "engine": "wav2vec2-onnx",
        "onnx_file": "onnx/model_quantized.onnx",
        "language_group": "da",
    },
    {
        "id": "roest-v3-315m",
        "name": "Røst v3 315M — CoRal v3 Danish Acoustic Specialist (FP16, ~631 MB)",
        "description": "3rd-generation Røst Danish Wav2Vec2 model trained on the newest CoRal-v3 conversational & dialect dataset in full 16-bit precision.",
        "size_mb": "631",
        "hf_repo": "MesterPaa/roest-v3-wav2vec2-315m-ONNX",
        "prefix": "roest-v3-315m",
        "engine": "wav2vec2-onnx",
        "onnx_file": "onnx/model_fp16.onnx",
        "language_group": "da",
    },
    {
        "id": "distil-large-v3.5",
        "name": "Distil-Whisper Large-v3.5 — English Specialist (Q4, ~696 MB)",
        "description": "Hugging Face's distilled English Whisper Large-v3.5 specialist: up to 6x faster than Large-v3 with ~99% accuracy on English speech.",
        "size_mb": "696",
        "hf_repo": "distil-whisper/distil-large-v3.5-ONNX",
        "prefix": "distil-large-v3.5",
        "engine": "hf-whisper-onnx",
        "encoder_file": "onnx/encoder_model_q4.onnx",
        "decoder_file": "onnx/decoder_model_merged_q4.onnx",
        "language_group": "en",
    },
    {
        "id": "kb-whisper-medium",
        "name": "KB-Whisper Medium — Swedish Specialist (KBLab Q4, ~652 MB)",
        "description": "National Library of Sweden (KBLab) Whisper specialist fine-tuned on massive Swedish broadcast, parliament, and conversational datasets.",
        "size_mb": "652",
        "hf_repo": "onnx-community/kb-whisper-medium-ONNX",
        "prefix": "kb-whisper-medium",
        "engine": "hf-whisper-onnx",
        "encoder_file": "onnx/encoder_model_q4.onnx",
        "decoder_file": "onnx/decoder_model_merged_q4.onnx",
        "language_group": "sv",
    },
    {
        "id": "nb-whisper-small",
        "name": "NB-Whisper Small — Norwegian Specialist (NbAiLab Q4, ~289 MB)",
        "description": "National Library of Norway (NbAiLab) Whisper specialist tuned specifically for Norwegian Bokmål and Nynorsk speech.",
        "size_mb": "289",
        "hf_repo": "onnx-community/nb-whisper-small-ONNX",
        "prefix": "nb-whisper-small",
        "engine": "hf-whisper-onnx",
        "encoder_file": "onnx/encoder_model_q4.onnx",
        "decoder_file": "onnx/decoder_model_merged_q4.onnx",
        "language_group": "no",
    },
    {
        "id": "whisper-turbo-de",
        "name": "Whisper Large-v3-Turbo German — German Specialist (Q4, ~728 MB)",
        "description": "Fast Large-v3-Turbo Whisper specialist fine-tuned specifically for German speech, compound words, and dialects.",
        "size_mb": "728",
        "hf_repo": "onnx-community/whisper-large-v3-turbo-german-ONNX",
        "prefix": "whisper-turbo-de",
        "engine": "hf-whisper-onnx",
        "encoder_file": "onnx/encoder_model_q4.onnx",
        "decoder_file": "onnx/decoder_model_merged_q4.onnx",
        "language_group": "de",
    },
    {
        "id": "whisper-es",
        "name": "Whisper Spanish Specialist (Q4, ~696 MB)",
        "description": "Fine-tuned Spanish Whisper ONNX specialist for accurate Spanish meetings, interviews, and regional accents.",
        "size_mb": "696",
        "hf_repo": "Juanpablozarza292/whisper_spanish_add-ONNX",
        "prefix": "whisper-es",
        "engine": "hf-whisper-onnx",
        "encoder_file": "onnx/encoder_model_q4.onnx",
        "decoder_file": "onnx/decoder_model_merged_q4.onnx",
        "language_group": "es",
    },
    {
        "id": "whisper-medium-fr",
        "name": "Whisper Medium French — French Specialist (Q4, ~652 MB)",
        "description": "Whisper Medium specialist fine-tuned specifically for French speech, accents, and punctuation.",
        "size_mb": "652",
        "hf_repo": "onnx-community/whisper-medium-fr-ONNX",
        "prefix": "whisper-medium-fr",
        "engine": "hf-whisper-onnx",
        "encoder_file": "onnx/encoder_model_q4.onnx",
        "decoder_file": "onnx/decoder_model_merged_q4.onnx",
        "language_group": "fr",
    },
    {
        "id": "whisper-small-it",
        "name": "Whisper Small Italian — Italian Specialist (Q4, ~289 MB)",
        "description": "Lightweight Whisper Small specialist fine-tuned specifically for fast, accurate Italian speech-to-text.",
        "size_mb": "289",
        "hf_repo": "onnx-community/whisper-small-ita-ONNX",
        "prefix": "whisper-small-it",
        "engine": "hf-whisper-onnx",
        "encoder_file": "onnx/encoder_model_q4.onnx",
        "decoder_file": "onnx/decoder_model_merged_q4.onnx",
        "language_group": "it",
    },
]

DEFAULT_WHISPER_MODEL = "small"

# Selectable 3rd Model Stage: Local AI Context Clarification & Word Validation Models
# Runs AFTER Speech-to-Text (Whisper) and BEFORE Translation to analyze what the conversation
# is about and validate/correct misheard words in each segment based on rolling context —
# without restructuring sentences.
CLARIFICATION_MODELS: List[Dict[str, str]] = [
    {
        "id": "none",
        "name": "Skip Clarification (Raw Speech-to-Text Output — 0 MB) — Default",
        "description": "Bypasses contextual word clarification and uses the raw first-pass Speech-to-Text transcription directly.",
        "size_mb": "0",
        "type": "none",
        "language_group": "all",
        "supported_languages": "all",
    },
    {
        "id": "whisper-context",
        "name": "Whisper Contextual Acoustic Validator (0 MB extra)",
        "description": "Validates and corrects misheard words using rolling conversation context (<|startofprev|>), topic keywords, and meeting spoken languages without restructuring sentences.",
        "size_mb": "0",
        "type": "acoustic-context",
        "language_group": "all",
        "supported_languages": "all",
    },
    {
        "id": "danskgpt-tiny",
        "name": "DanskGPT Tiny Chat — Danish Specialist (ONNX INT8, ~1.05 GB)",
        "description": "Danish-specialized local LLM fine-tuned for natural Danish spelling, compound words, punctuation, and context-aware word correction.",
        "size_mb": "1051",
        "type": "qwen-onnx",
        "hf_repo": "varsan-g/danskgpt-tiny-chat-onnx",
        "onnx_file": "onnx/model_int8.onnx",
        "num_layers": "22",
        "num_kv_heads": "4",
        "head_dim": "64",
        "chat_template": "chatml",
        "language_group": "da",
        "supported_languages": "da",
    },
    {
        "id": "eurollm-1.7b",
        "name": "EuroLLM 1.7B Instruct — European 24-Language Specialist (~1.58 GB)",
        "description": "Purpose-built European LLM trained across all 24 EU languages (Danish, Swedish, Norwegian, German, Spanish, French, Italian & English) for accurate European grammar and domain terms.",
        "size_mb": "1582",
        "type": "qwen-onnx",
        "hf_repo": "flackzz/EuroLLM-1.7B-Instruct-ONNX",
        "onnx_file": "onnx/model_quantized.onnx",
        "num_layers": "24",
        "num_kv_heads": "8",
        "head_dim": "128",
        "chat_template": "chatml",
        "language_group": "eu",
        "supported_languages": "da,sv,no,de,es,fr,it,en,eu",
    },
    {
        "id": "smollm2-360m",
        "name": "SmolLM2 360M Instruct — English Ultra-Fast Specialist (Q4, ~369 MB)",
        "description": "Compact, ultra-fast English instruction model ideal for quick English spelling, homophone, and punctuation clarification on any laptop.",
        "size_mb": "369",
        "type": "qwen-onnx",
        "hf_repo": "onnx-community/SmolLM2-360M-Instruct-ONNX",
        "onnx_file": "onnx/model_q4.onnx",
        "num_layers": "32",
        "num_kv_heads": "5",
        "head_dim": "64",
        "chat_template": "chatml",
        "language_group": "en",
        "supported_languages": "en",
    },
    {
        "id": "llama-3.2-1b",
        "name": "Llama 3.2 1B Instruct — Western EU & English Specialist (INT8, ~1.18 GB)",
        "description": "Meta Llama 3.2 1B tuned for English, German, French, Italian, and Spanish — strong sentence-preserving domain word & grammar validation.",
        "size_mb": "1179",
        "type": "qwen-onnx",
        "hf_repo": "onnx-community/Llama-3.2-1B-Instruct-ONNX",
        "onnx_file": "onnx/model_int8.onnx",
        "num_layers": "16",
        "num_kv_heads": "8",
        "head_dim": "64",
        "chat_template": "llama3",
        "language_group": "west-eu",
        "supported_languages": "en,de,fr,es,it,west-eu",
    },
    {
        "id": "llama-3.2-3b",
        "name": "Llama 3.2 3B Instruct — High-Precision Western EU & EN (Q4F16, ~2.3 GB)",
        "description": "Larger Meta Llama 3.2 3B model for deep domain-term disambiguation and grammar accuracy in English, German, French, Spanish, and Italian.",
        "size_mb": "2296",
        "type": "qwen-onnx",
        "hf_repo": "onnx-community/Llama-3.2-3B-Instruct-ONNX",
        "onnx_file": "onnx/model_q4f16.onnx",
        "onnx_extra_files": "onnx/model_q4f16.onnx_data,onnx/model_q4f16.onnx_data_1",
        "num_layers": "28",
        "num_kv_heads": "8",
        "head_dim": "128",
        "chat_template": "llama3",
        "language_group": "west-eu",
        "supported_languages": "en,de,fr,es,it,west-eu",
    },
    {
        "id": "phi-4-mini",
        "name": "Microsoft Phi-4 Mini Instruct — Technical & EU Specialist (Q4F16, ~2.4 GB)",
        "description": "Microsoft's 3.8B reasoning-tuned multilingual model (23 languages incl. DA, SV, NO, DE, ES, FR, IT, EN) — excels at technical, medical, and business terminology.",
        "size_mb": "2433",
        "type": "qwen-onnx",
        "hf_repo": "onnx-community/Phi-4-mini-instruct-ONNX",
        "onnx_file": "onnx/model_q4f16.onnx",
        "onnx_extra_files": "onnx/model_q4f16.onnx_data,onnx/model_q4f16.onnx_data_1",
        "num_layers": "32",
        "num_kv_heads": "8",
        "head_dim": "128",
        "chat_template": "phi4",
        "language_group": "eu",
        "supported_languages": "da,sv,no,de,es,fr,it,en,eu,multilingual",
    },
    {
        "id": "gemma-3-270m",
        "name": "Gemma 3 270M Instruct — Ultra-Light Nordic & EU (Q4, ~308 MB)",
        "description": "Ultra-compact Google Gemma 3 model (140+ languages) for fast, low-RAM punctuation and context fixes across Danish, Swedish, Norwegian, and EU languages.",
        "size_mb": "308",
        "type": "qwen-onnx",
        "hf_repo": "onnx-community/gemma-3-270m-it-ONNX",
        "onnx_file": "onnx/model_q4.onnx",
        "onnx_extra_files": "onnx/model_q4.onnx_data",
        "num_layers": "18",
        "num_kv_heads": "1",
        "head_dim": "256",
        "chat_template": "gemma",
        "language_group": "multilingual",
        "supported_languages": "da,sv,no,de,es,fr,it,en,eu,multilingual",
    },
    {
        "id": "gemma-3-1b",
        "name": "Gemma 3 1B Instruct — Nordic & Multilingual Specialist (INT8, ~955 MB)",
        "description": "Google Gemma 3 1B supporting 140+ languages with strong Nordic (DA/SV/NO) and European punctuation and contextual word accuracy.",
        "size_mb": "955",
        "type": "qwen-onnx",
        "hf_repo": "onnx-community/gemma-3-1b-it-ONNX",
        "onnx_file": "onnx/model_int8.onnx",
        "num_layers": "26",
        "num_kv_heads": "1",
        "head_dim": "256",
        "chat_template": "gemma",
        "language_group": "multilingual",
        "supported_languages": "da,sv,no,de,es,fr,it,en,eu,multilingual",
    },
    {
        "id": "qwen3-0.6b",
        "name": "Qwen 3 0.6B — Next-Gen Fast Multilingual Clarifier (INT8, ~589 MB)",
        "description": "Fast next-generation Qwen 3 model supporting 119 languages for quick context-aware spelling and punctuation validation.",
        "size_mb": "589",
        "type": "qwen-onnx",
        "hf_repo": "onnx-community/Qwen3-0.6B-ONNX",
        "onnx_file": "onnx/model_int8.onnx",
        "num_layers": "28",
        "num_kv_heads": "8",
        "head_dim": "128",
        "chat_template": "chatml",
        "language_group": "multilingual",
        "supported_languages": "da,sv,no,de,es,fr,it,en,eu,multilingual",
    },
    {
        "id": "qwen3-1.7b",
        "name": "Qwen 3 1.7B — Next-Gen Deep Multilingual Clarifier (INT8, ~1.66 GB)",
        "description": "High-accuracy next-generation Qwen 3 1.7B model (119 languages) for deep contextual word correction across Nordic, European, and global meetings.",
        "size_mb": "1662",
        "type": "qwen-onnx",
        "hf_repo": "onnx-community/Qwen3-1.7B-ONNX",
        "onnx_file": "onnx/model_int8.onnx",
        "num_layers": "28",
        "num_kv_heads": "8",
        "head_dim": "128",
        "chat_template": "chatml",
        "language_group": "multilingual",
        "supported_languages": "da,sv,no,de,es,fr,it,en,eu,multilingual",
    },
    {
        "id": "qwen2.5-0.5b",
        "name": "Qwen 2.5 0.5B Context Clarifier (ONNX Q4, ~755 MB)",
        "description": "Lightweight multilingual causal LLM that validates phonetically misheard words against conversation context with strict sentence-structure preservation.",
        "size_mb": "755",
        "type": "qwen-onnx",
        "hf_repo": "onnx-community/Qwen2.5-0.5B-Instruct",
        "onnx_file": "onnx/model_q4.onnx",
        "num_layers": "24",
        "num_kv_heads": "2",
        "head_dim": "64",
        "chat_template": "chatml",
        "language_group": "multilingual",
        "supported_languages": "en,de,es,fr,it,multilingual",
    },
    {
        "id": "qwen2.5-1.5b",
        "name": "Qwen 2.5 1.5B Deep Context Clarifier (ONNX Q4, ~1.7 GB)",
        "description": "Larger 1.5B multilingual context validator for deeper domain & Danish/English compound word correction without altering sentence structure.",
        "size_mb": "1705",
        "type": "qwen-onnx",
        "hf_repo": "onnx-community/Qwen2.5-1.5B-Instruct",
        "onnx_file": "onnx/model_q4.onnx",
        "num_layers": "28",
        "num_kv_heads": "2",
        "head_dim": "128",
        "chat_template": "chatml",
        "language_group": "multilingual",
        "supported_languages": "da,en,de,es,fr,it,multilingual",
    },
]

DEFAULT_CLARIFICATION_MODEL = "none"

# Selectable Local AI Translation Models (Focused on Danish <-> English + Spanish/German/Multilingual)
TRANSLATION_MODELS: List[Dict[str, str]] = [
    {
        "id": "opus-mt-da-en",
        "name": "OPUS-MT Danish ↔ English Specialist (~85 MB)",
        "description": "Specialized Helsinki-NLP MarianMT neural translator tuned specifically for Danish ↔ English (plus Spanish/German pairs).",
        "size_mb": "85",
        "type": "opus",
    },
    {
        "id": "nllb-200-600M",
        "name": "Meta NLLB-200 Universal (200 Languages, ~580 MB)",
        "description": "Heavyweight 200-language neural machine translator for any-to-any language combinations.",
        "size_mb": "580",
        "type": "nllb",
    },
    {
        "id": "whisper-native",
        "name": "Whisper Direct Audio-to-English Translator (0 MB extra)",
        "description": "Uses Whisper's built-in acoustic translation head to translate speech directly into English.",
        "size_mb": "0",
        "type": "whisper",
    },
    {
        "id": "none",
        "name": "Skip Translation — Transcription Only (0 MB) — Default",
        "description": "Disables translation completely and keeps the transcript in each speaker's original spoken language.",
        "size_mb": "0",
        "type": "none",
    },
]

DEFAULT_TRANSLATION_MODEL = "none"

# Specialized OPUS-MT ONNX repositories per language pair (including direct EU pairs)
OPUS_PAIR_REPOS: Dict[str, str] = {
    "da-en": "Xenova/opus-mt-da-en",
    "en-da": "Xenova/opus-mt-en-da",
    "es-en": "Xenova/opus-mt-es-en",
    "en-es": "Xenova/opus-mt-en-es",
    "de-en": "Xenova/opus-mt-de-en",
    "en-de": "Xenova/opus-mt-en-de",
    "fr-en": "Xenova/opus-mt-fr-en",
    "en-fr": "Xenova/opus-mt-en-fr",
    "it-en": "Xenova/opus-mt-it-en",
    "en-it": "Xenova/opus-mt-en-it",
    "sv-en": "Xenova/opus-mt-sv-en",
    "en-sv": "Xenova/opus-mt-en-sv",
    "no-en": "Xenova/opus-mt-da-en",  # Norwegian Bokmål shares vocabulary with Danish-English OPUS
    "en-no": "Xenova/opus-mt-en-da",
    # Direct European language pairs (no English pivot needed!)
    "da-de": "Xenova/opus-mt-da-de",
    "de-fr": "Xenova/opus-mt-de-fr",
    "fr-de": "Xenova/opus-mt-fr-de",
    "de-es": "Xenova/opus-mt-de-es",
    "es-de": "Xenova/opus-mt-es-de",
    "fr-es": "Xenova/opus-mt-fr-es",
    "es-fr": "Xenova/opus-mt-es-fr",
    "it-fr": "Xenova/opus-mt-it-fr",
    "es-it": "Xenova/opus-mt-es-it",
    "it-es": "Xenova/opus-mt-it-es",
}

# NLLB-200 ONNX repository for 100% offline any-to-any translation
NLLB_REPO_ID = "Xenova/nllb-200-distilled-600M"
NLLB_LOCAL_DIR = MODELS_DIR / "nllb-200-onnx-int8"

# Selectable Step 1: SpeechSplitter / Voice Diarization ONNX Models (Sherpa-ONNX 80-dim Fbank)
_SPEAKER_RELEASE_BASE = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
)
SPEAKER_DIARIZATION_MODELS: List[Dict[str, str]] = [
    {
        "id": "campplus-voxceleb",
        "name": "CAM++ VoxCeleb — Western/EU Fast & Accurate (~28 MB)",
        "description": "Fast, lightweight speaker voice-embedding model trained on Western & European VoxCeleb voices. Recommended default for crisp 2–4 speaker separation.",
        "size_mb": "28",
        "filename": "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
        "url": _SPEAKER_RELEASE_BASE + "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
        "type": "onnx",
    },
    {
        "id": "wespeaker-resnet293",
        "name": "WeSpeaker ResNet293-LM — High-Precision Studio (~109 MB)",
        "description": "Top-tier high-precision VoxCeleb voice embedding model for challenging meetings with 3+ similar voices or background room echo.",
        "size_mb": "109",
        "filename": "wespeaker_en_voxceleb_resnet293_LM.onnx",
        "url": _SPEAKER_RELEASE_BASE + "wespeaker_en_voxceleb_resnet293_LM.onnx",
        "type": "onnx",
    },
    {
        "id": "3dspeaker-eres2net",
        "name": "3D-Speaker ERes2Net Base (~38 MB)",
        "description": "Classic 3D-Speaker ERes2Net acoustic embedding model for balanced multi-speaker clustering across international voices.",
        "size_mb": "38",
        "filename": "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
        "url": _SPEAKER_RELEASE_BASE + "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
        "type": "onnx",
    },
    {
        "id": "none",
        "name": "Single-Speaker Mode — Skip Diarization (0 MB)",
        "description": "Disables speaker clustering and assigns all speech to Speaker 1. Ideal for solo dictations, voice notes, or 1-person lectures.",
        "size_mb": "0",
        "filename": "",
        "url": "",
        "type": "none",
    },
]
DEFAULT_DIARIZATION_MODEL = "campplus-voxceleb"

# Legacy aliases kept for backwards compatibility
SPEAKER_EMBEDDING_MODEL_URL = (
    _SPEAKER_RELEASE_BASE
    + "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
)
SPEAKER_EMBEDDING_MODEL_PATH = (
    MODELS_DIR / "speaker_embedding" / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
)

# Comprehensive mapping of Whisper language codes -> Display Names and NLLB-200 FLORES-200 codes
SUPPORTED_LANGUAGES: List[Dict[str, str]] = [
    {"code": "auto", "name": "Auto-Detect Language", "nllb": ""},
    {"code": "none", "name": "Skip Translation (Keep Spoken Language)", "nllb": ""},
    {"code": "da", "name": "Danish (Dansk)", "nllb": "dan_Latn"},
    {"code": "en", "name": "English", "nllb": "eng_Latn"},
    {"code": "es", "name": "Spanish (Español)", "nllb": "spa_Latn"},
    {"code": "de", "name": "German (Deutsch)", "nllb": "deu_Latn"},
    {"code": "fr", "name": "French (Français)", "nllb": "fra_Latn"},
    {"code": "sv", "name": "Swedish (Svenska)", "nllb": "swe_Latn"},
    {"code": "no", "name": "Norwegian (Norsk)", "nllb": "nob_Latn"},
    {"code": "nl", "name": "Dutch (Nederlands)", "nllb": "nld_Latn"},
    {"code": "it", "name": "Italian (Italiano)", "nllb": "ita_Latn"},
    {"code": "pt", "name": "Portuguese (Português)", "nllb": "por_Latn"},
    {"code": "pl", "name": "Polish (Polski)", "nllb": "pol_Latn"},
    {"code": "fi", "name": "Finnish (Suomi)", "nllb": "fin_Latn"},
    {"code": "uk", "name": "Ukrainian (Українська)", "nllb": "ukr_Cyrl"},
    {"code": "ru", "name": "Russian (Русский)", "nllb": "rus_Cyrl"},
    {"code": "tr", "name": "Turkish (Türkçe)", "nllb": "tur_Latn"},
    {"code": "ar", "name": "Arabic (العربية)", "nllb": "arb_Arab"},
    {"code": "hi", "name": "Hindi (हिन्दी)", "nllb": "hin_Deva"},
    {"code": "zh", "name": "Chinese (中文)", "nllb": "zho_Hans"},
    {"code": "ja", "name": "Japanese (日本語)", "nllb": "jpn_Jpan"},
    {"code": "ko", "name": "Korean (한국어)", "nllb": "kor_Hang"},
    {"code": "cs", "name": "Czech (Čeština)", "nllb": "ces_Latn"},
    {"code": "ro", "name": "Romanian (Română)", "nllb": "ron_Latn"},
    {"code": "hu", "name": "Hungarian (Magyar)", "nllb": "hun_Latn"},
    {"code": "el", "name": "Greek (Ελληνικά)", "nllb": "ell_Grek"},
]

LANGUAGE_NAME_MAP: Dict[str, str] = {lang["code"]: lang["name"] for lang in SUPPORTED_LANGUAGES}
LANGUAGE_NLLB_MAP: Dict[str, str] = {lang["code"]: lang["nllb"] for lang in SUPPORTED_LANGUAGES if lang["nllb"]}

# Default speaker badge colors for UI
SPEAKER_COLORS: List[str] = [
    "#3B82F6",  # Blue (Speaker 1)
    "#10B981",  # Emerald (Speaker 2)
    "#F59E0B",  # Amber (Speaker 3)
    "#EC4899",  # Pink (Speaker 4)
    "#8B5CF6",  # Purple (Speaker 5)
    "#06B6D4",  # Cyan (Speaker 6)
    "#F97316",  # Orange (Speaker 7)
    "#14B8A6",  # Teal (Speaker 8)
]
