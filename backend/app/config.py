import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

FROZEN = getattr(sys, "frozen", False)
# Bundled read-only resources (frontend/) vs. writable user data (data/, .env, models/)
RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
PROJECT_ROOT = Path(sys.executable).parent if FROZEN else Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Classroom Face App"
    camera_source: str = "0"
    camera_backend: str = "auto"
    camera_width: int = 2560
    camera_height: int = 1440
    camera_fps: int = 15
    processing_width: int = 1280
    processing_height: int = 720
    processing_fps: int = 6
    recognition_min_quality: float = 0.55
    recognition_min_similarity: float = 0.45
    data_dir: Path = PROJECT_ROOT / "data"
    model_dir: Path = PROJECT_ROOT / "models"

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
