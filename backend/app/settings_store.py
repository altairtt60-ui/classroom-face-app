"""Small JSON file for choices made in the UI (camera source, etc.) so nobody has to edit .env."""
from __future__ import annotations

import json
from threading import Lock

from .config import settings

_lock = Lock()


def _path():
    return settings.data_dir / "settings.json"


def load_into_settings() -> None:
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    for key in ("camera_source", "camera_backend", "camera_width", "camera_height", "camera_fps"):
        if key in data:
            setattr(settings, key, data[key])


def save_camera_choice(source: str, backend: str) -> None:
    with _lock:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        data.update({"camera_source": source, "camera_backend": backend})
        _path().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
