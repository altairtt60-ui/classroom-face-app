import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

FROZEN = getattr(sys, "frozen", False)
# Bundled read-only resources (frontend/) vs. writable user data (data/, .env, models/)
RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
PROJECT_ROOT = Path(sys.executable).parent if FROZEN else Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Classroom Face App"

    # Camera. Usually chosen from the UI (saved to data/settings.json); these are just defaults.
    camera_source: str = "0"
    camera_backend: str = "auto"
    camera_width: int = 1920
    camera_height: int = 1080
    camera_fps: int = 15

    # Vision
    processing_fps: float = 4.0
    yolo_model: str = "yolo11n.pt"
    yolo_imgsz: int = 960
    yolo_conf: float = 0.35
    stream_max_width: int = 1280

    # Recognition / identity memory
    recognition_min_quality: float = 0.35
    recognition_min_similarity: float = 0.38
    recognition_min_margin: float = 0.05
    recognition_min_votes: int = 2
    probe_seconds: float = 1.2
    reverify_seconds: float = 20.0
    track_identity_ttl_seconds: float = 120.0
    relink_seconds: float = 45.0

    data_dir: Path = PROJECT_ROOT / "data"
    model_dir: Path = PROJECT_ROOT / "models"

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
