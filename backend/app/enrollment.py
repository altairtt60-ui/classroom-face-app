"""Turning an uploaded photo into a stored face template."""
from __future__ import annotations

import secrets
from pathlib import Path

import cv2
import numpy as np

from .config import settings
from .face_engine import engine


class EnrollmentError(ValueError):
    """Message is shown to the user as-is, so keep it in Kazakh and specific."""


MAX_SIDE = 1600  # downscaling large phone photos makes detection faster and more reliable


def _decode(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise EnrollmentError("Жүктелген файл жарамды сурет емес")
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest > MAX_SIDE:
        scale = MAX_SIDE / longest
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image


def save_photo_file(data: bytes) -> tuple[Path, str]:
    photo_dir = settings.data_dir / "cadets"
    photo_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{secrets.token_hex(8)}.jpg"
    path = photo_dir / filename
    path.write_bytes(data)
    return path, filename


def build_template(data: bytes) -> tuple[str, np.ndarray, float]:
    """Validate the photo and return (saved filename, embedding, quality) for one clear face.

    Raises EnrollmentError with a message that tells the person what to fix, instead of a
    bare "invalid" - a blocking wall of unexplained 422s is what made this feel broken.
    """
    image = _decode(data)
    try:
        faces = engine.analyze(image)
    except RuntimeError as exc:
        raise EnrollmentError(str(exc)) from exc
    if not faces:
        raise EnrollmentError(
            "Суретте бет табылмады. Жарық жақсы, бет камераға қарап тұрған, бүкіл бас көрінетін фото жүктеңіз."
        )
    if len(faces) > 1:
        raise EnrollmentError("Суретте бірнеше бет табылды. Тек сол курсанттың жеке фотосын жүктеңіз.")
    face = faces[0]
    if face.quality < settings.recognition_min_quality:
        raise EnrollmentError(
            "Фото сапасы төмен (бұлыңғыр, тым кіші немесе бет қисық тұр). Жақыннан, анық түсірілген фото жүктеп көріңіз."
        )
    photo_path, filename = save_photo_file(data)
    return filename, face.embedding, face.quality
