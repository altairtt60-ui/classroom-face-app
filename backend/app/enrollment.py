from pathlib import Path

import cv2
import numpy as np

from .config import settings


class EnrollmentError(ValueError):
    pass


def save_photo(data: bytes, student_code: str) -> Path:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    photo_dir = settings.data_dir / "cadets"
    photo_dir.mkdir(parents=True, exist_ok=True)
    path = photo_dir / f"{student_code}.jpg"
    path.write_bytes(data)
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        path.unlink(missing_ok=True)
        raise EnrollmentError("Жүктелген файл жарамды сурет емес")
    return path


def create_embedding(photo_path: Path, student_code: str) -> Path | None:
    """Create an InsightFace embedding when the optional dependency is installed.

    The app remains usable for profile management before the recognition models are installed.
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        return None

    image = cv2.imread(str(photo_path))
    if image is None:
        raise EnrollmentError("Суретті оқу мүмкін болмады")

    analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    analyzer.prepare(ctx_id=0, det_size=(640, 640))
    faces = analyzer.get(image)
    if len(faces) != 1:
        raise EnrollmentError("Суретте дәл бір бет болуы керек")

    embedding_dir = settings.data_dir / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = embedding_dir / f"{student_code}.npy"
    np.save(embedding_path, faces[0].embedding.astype(np.float32))
    return embedding_path
