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


def _load_initial_data_dir() -> Path:
    if SETTINGS_FILE.exists():
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            custom = data.get("storage_path", "").strip()
            if custom:
                p = Path(custom).resolve()
                p.mkdir(parents=True, exist_ok=True)
                return p
        except Exception:
            pass
    return BASE_DIR / "data"


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

    SETTINGS_FILE.write_text(
        json.dumps({"storage_path": str(DATA_DIR)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
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
    },
    {
        "id": "base",
        "name": "Whisper Base (~195 MB)",
        "description": "Fast & lightweight for clear speech.",
        "size_mb": "195",
        "hf_repo": "csukuangfj/sherpa-onnx-whisper-base",
        "prefix": "base",
    },
    {
        "id": "small",
        "name": "Whisper Small (~610 MB) — Recommended for Danish",
        "description": "Great balance of live speed and strong Danish, Spanish, German & English accuracy.",
        "size_mb": "610",
        "hf_repo": "csukuangfj/sherpa-onnx-whisper-small",
        "prefix": "small",
    },
    {
        "id": "medium",
        "name": "Whisper Medium (~1.7 GB) — High Accuracy Danish",
        "description": "High accuracy for Danish dialects, strong accents, and noisy meetings.",
        "size_mb": "1700",
        "hf_repo": "csukuangfj/sherpa-onnx-whisper-medium",
        "prefix": "medium",
    },
    {
        "id": "turbo",
        "name": "Whisper Large-v3-Turbo (~1.6 GB) — Best Danish Quality",
        "description": "Tip-top Large-v3 Danish & multilingual accuracy at fast real-time speed.",
        "size_mb": "1600",
        "hf_repo": "csukuangfj/sherpa-onnx-whisper-turbo",
        "prefix": "turbo",
    },
]

DEFAULT_WHISPER_MODEL = "small"

# Selectable Local AI Translation Models (Focused on Danish <-> English + Spanish/German/Multilingual)
TRANSLATION_MODELS: List[Dict[str, str]] = [
    {
        "id": "opus-mt-da-en",
        "name": "OPUS-MT Danish ↔ English Specialist (~85 MB) — Default",
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
]

DEFAULT_TRANSLATION_MODEL = "opus-mt-da-en"

# Specialized OPUS-MT ONNX repositories per language pair
OPUS_PAIR_REPOS: Dict[str, str] = {
    "da-en": "Xenova/opus-mt-da-en",
    "en-da": "Xenova/opus-mt-en-da",
    "es-en": "Xenova/opus-mt-es-en",
    "en-es": "Xenova/opus-mt-en-es",
    "de-en": "Xenova/opus-mt-de-en",
    "en-de": "Xenova/opus-mt-en-de",
    "fr-en": "Xenova/opus-mt-fr-en",
    "en-fr": "Xenova/opus-mt-en-fr",
    "sv-en": "Xenova/opus-mt-sv-en",
    "no-en": "Xenova/opus-mt-da-en",  # Norwegian Bokmål shares vocabulary with Danish-English OPUS
}

# NLLB-200 ONNX repository for 100% offline any-to-any translation
NLLB_REPO_ID = "Xenova/nllb-200-distilled-600M"
NLLB_LOCAL_DIR = MODELS_DIR / "nllb-200-onnx-int8"

# Speaker Embedding ONNX model (Sherpa-ONNX 3D-Speaker / ERes2Net)
SPEAKER_EMBEDDING_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
    "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
)
SPEAKER_EMBEDDING_MODEL_PATH = (
    MODELS_DIR / "speaker_embedding" / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
)

# Comprehensive mapping of Whisper language codes -> Display Names and NLLB-200 FLORES-200 codes
SUPPORTED_LANGUAGES: List[Dict[str, str]] = [
    {"code": "auto", "name": "Auto-Detect Language", "nllb": ""},
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
