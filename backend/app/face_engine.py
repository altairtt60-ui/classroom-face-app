"""One shared InsightFace instance, loaded once for the whole app (enrollment AND live
recognition). Loading it per-request, like the original code did, adds ~60-90s to every
request that touches faces — that was the real cause of the app "hanging" when adding
a cadet.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FaceObservation:
    embedding: np.ndarray  # unit length
    bbox: tuple[float, float, float, float]
    det_score: float
    quality: float  # 0..1 combined score
    face_px: float  # face width in pixels
    blur: float
    yaw: float


def compute_quality(det_score: float, face_px: float, blur: float, yaw: float) -> float:
    """Detector confidence x face size x sharpness x head-turn, each 0..1, multiplied so a
    face must be reasonable on every axis (a big blurry face shouldn't outscore a small sharp one)."""
    size_factor = min(1.0, face_px / 60.0)
    blur_factor = min(1.0, blur / 60.0)
    pose_factor = max(0.0, 1.0 - abs(yaw) / 70.0)
    return float(max(0.0, min(1.0, det_score)) * size_factor * blur_factor * pose_factor)


class FaceEngine:
    def __init__(self) -> None:
        self._app = None
        self._lock = threading.Lock()
        self._det_size: tuple[int, int] | None = None
        self.load_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._app is not None

    def _load(self):
        if self._app is not None:
            return self._app
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            self.load_error = "InsightFace орнатылмаған (pip install insightface onnxruntime)"
            raise RuntimeError(self.load_error) from exc
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "landmark_3d_68", "recognition"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        self._app, self._det_size, self.load_error = app, (640, 640), None
        return app

    def warmup(self) -> None:
        """Call once at startup (in a background thread) so the first real request is fast."""
        with self._lock:
            try:
                self._load()
            except RuntimeError:
                pass

    def analyze(self, image: np.ndarray, det_size: int = 640) -> list[FaceObservation]:
        """Detect every face in `image` and compute a unit embedding + quality score for each."""
        with self._lock:
            app = self._load()
            if self._det_size != (det_size, det_size):
                app.det_model.input_size = (det_size, det_size)
                self._det_size = (det_size, det_size)
            faces = app.get(image)
        gray = None
        h, w = image.shape[:2]
        results: list[FaceObservation] = []
        for face in faces:
            x1, y1, x2, y2 = [float(v) for v in face.bbox]
            ix1, iy1 = max(0, int(x1)), max(0, int(y1))
            ix2, iy2 = min(w, int(x2)), min(h, int(y2))
            if ix2 - ix1 < 4 or iy2 - iy1 < 4:
                continue
            if gray is None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            patch = cv2.resize(gray[iy1:iy2, ix1:ix2], (64, 64))
            blur = float(cv2.Laplacian(patch, cv2.CV_64F).var())
            pose = getattr(face, "pose", None)
            yaw = float(pose[1]) if pose is not None else 0.0
            vector = np.asarray(face.embedding, dtype=np.float32)
            norm = np.linalg.norm(vector)
            if not norm:
                continue
            det = float(face.det_score)
            results.append(
                FaceObservation(
                    embedding=vector / norm,
                    bbox=(x1, y1, x2, y2),
                    det_score=det,
                    quality=compute_quality(det, x2 - x1, blur, yaw),
                    face_px=x2 - x1,
                    blur=blur,
                    yaw=yaw,
                )
            )
        return results

    def best_face(self, image: np.ndarray) -> FaceObservation | None:
        faces = self.analyze(image)
        return max(faces, key=lambda f: f.quality) if faces else None


engine = FaceEngine()
