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
ROOT_PATH = os.getenv("ROOT_PATH", "")

_WINGET_FFMPEG = Path.home() / "AppData/Local/Microsoft/WinGet/Links/ffmpeg.exe"
FFMPEG = shutil.which("ffmpeg") or (str(_WINGET_FFMPEG) if _WINGET_FFMPEG.exists() else "ffmpeg")
