from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

from .config import settings


@dataclass
class Match:
    cadet_id: int | None
    full_name: str | None
    similarity: float


class RecognitionService:
    def __init__(self) -> None:
        self._app = None
        self._templates: list[tuple[int, str, np.ndarray]] = []
        self._lock = Lock()

    def _load_model(self):
        if self._app is None:
            try:
                from insightface.app import FaceAnalysis
            except ImportError as exc:
                raise RuntimeError("InsightFace орнатылмаған. pip install insightface onnxruntime") from exc
            self._app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            self._app.prepare(ctx_id=0, det_size=(640, 640))
        return self._app

    def embed(self, image: np.ndarray) -> np.ndarray:
        faces = self._load_model().get(image)
        if len(faces) != 1:
            raise ValueError("Recognition crop ішінде дәл бір бет болуы керек")
        vector = faces[0].embedding.astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def load_templates(self, cadets: list[dict]) -> None:
        templates: list[tuple[int, str, np.ndarray]] = []
        for cadet in cadets:
            path_value = cadet.get("embedding_path")
            if not path_value:
                continue
            path = Path(path_value)
            if path.exists():
                vector = np.load(path).astype(np.float32)
                norm = np.linalg.norm(vector)
                if norm:
                    templates.append((int(cadet["id"]), cadet["full_name"], vector / norm))
        with self._lock:
            self._templates = templates

    def match(self, embedding: np.ndarray, min_similarity: float | None = None) -> list[Match]:
        threshold = min_similarity if min_similarity is not None else settings.recognition_min_similarity
        with self._lock:
            templates = list(self._templates)
        if not templates:
            return []
        results = [
            Match(cadet_id, name, float(np.dot(embedding, vector)))
            for cadet_id, name, vector in templates
        ]
        results.sort(key=lambda item: item.similarity, reverse=True)
        return [item for item in results if item.similarity >= threshold]

    def recognize_crop(self, crop: np.ndarray) -> list[Match]:
        return self.match(self.embed(crop))
