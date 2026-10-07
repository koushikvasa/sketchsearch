import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

try:  # Norton intercepts TLS on the dev machine; trust the OS certificate store.
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"
SEGMENTS_DIR = DATA_DIR / "segments"
FRAMES_DIR = DATA_DIR / "frames"
YOLO_RAW_DIR = DATA_DIR / "yolo_raw"
GT_FILE = DATA_DIR / "raw" / "warehouse" / "ground_truth.json"
ALIAS_REPORT = DATA_DIR / "alias_report.json"
CLIP_EMBEDDINGS = DATA_DIR / "clip_embeddings.npy"
CLIP_IDS = DATA_DIR / "clip_ids.json"
INDEX_FILES = {"gt": DATA_DIR / "index.json", "yolo": DATA_DIR / "index_yolo.json"}

SOURCE = os.getenv("SOURCE", "local")
INDEX_SOURCE = os.getenv("INDEX_SOURCE", "gt")
TEXT_PREFILTER = os.getenv("TEXT_PREFILTER", "false").strip().lower() in ("1", "true", "yes", "on")
ROOT_PATH = os.getenv("ROOT_PATH", "")
WEAVE_PROJECT = os.getenv("WEAVE_PROJECT", "sketchsearch")
VERIFY_CACHE_DIR = DATA_DIR / "verify_cache"

# Text LLM (words -> sketch, agent): any OpenAI-compatible endpoint, picked by LLM_PROVIDER.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
LLM_PROVIDERS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY",
             "model_env": "GROQ_MODEL", "default_model": "openai/gpt-oss-120b", "json_mode": True},
    # W&B's JSON mode garbles gpt-oss output (tested 2026-10-07); plain text + low reasoning is valid in ~10 s.
    "wandb": {"base_url": "https://api.inference.wandb.ai/v1", "key_env": "WANDB_API_KEY",
              "model_env": "WANDB_MODEL", "default_model": "openai/gpt-oss-120b", "json_mode": False,
              "extra": {"reasoning_effort": "low"}},
}

# Vision (verification, diagram -> sketch).
VISION_PROVIDER = os.getenv("VISION_PROVIDER", "gemini").strip().lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or "gemini-3.5-flash-lite"
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "")
VERIFY_TIMEOUT_S = 20
VERIFY_CONCURRENCY = 3

# Sketch label -> track labels of the YOLO (COCO) index, from data/alias_report.json. YOLO11 never
# detected the forklift / robot / transporter (the trucks/boats it reported were shelving), so those
# map to nothing. The GT index already uses sketch labels, so it needs no aliases.
LABEL_ALIASES: dict[str, dict[str, list[str]]] = {
    "gt": {},
    "yolo": {"person": ["person"], "forklift": [], "robot": [], "transporter": []},
}

_WINGET_FFMPEG = Path.home() / "AppData/Local/Microsoft/WinGet/Links/ffmpeg.exe"
FFMPEG = shutil.which("ffmpeg") or (str(_WINGET_FFMPEG) if _WINGET_FFMPEG.exists() else "ffmpeg")
