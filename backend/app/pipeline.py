"""Per-frame: detect+track people (YOLO/ByteTrack), then, for tracks that need it, crop the
head area, run face recognition, and vote an identity onto that track via TrackIdentityCache.

Why this way and not "recognize every face every frame":
- A classroom camera looks down from above the board; most of a session, tracked people
  barely move, so recognizing every frame is wasted work and it means a person who turns
  away or is briefly hidden instantly "loses" their name. Voting + memory (identity_cache.py)
  means a name sticks with a track id, survives a few bad frames, and comes back on the
  SAME name if the tracker drops and re-acquires the person.

Two hard-won lessons are baked into this file, because both of them once made the live view
show a perfectly good camera picture with no boxes and no names:
- the custom ByteTrack settings file must exist (see tracker_config(), which falls back to
  the built-in tracker instead of raising on every frame);
- the label is drawn with overlay_text.draw_label(), not cv2.putText(), because the Hershey
  fonts cannot draw Kazakh/Cyrillic names.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import cv2
import numpy as np

from .config import RESOURCE_ROOT, settings
from .face_engine import engine as face_engine
from .identity_cache import TrackIdentityCache
from .overlay_text import draw_label
from .recognition import RecognitionService

# Custom ByteTrack settings (long lost-track buffer, so a name survives someone being hidden
# behind a neighbour) shipped next to the app; ultralytics' built-in config is the fallback.
TRACKER_FILE = RESOURCE_ROOT / "backend" / "app" / "trackers" / "classroom_bytetrack.yaml"
DEFAULT_TRACKER = "bytetrack.yaml"


def tracker_config() -> str:
    """Path to the custom tracker settings, or ultralytics' built-in one if it is missing.

    Shipping the YAML under the wrong name used to make *every* model.track() call raise
    FileNotFoundError; the error was swallowed and the stream therefore showed no detections
    at all. Degrading to the built-in tracker keeps the app usable in that situation.
    """
    return str(TRACKER_FILE) if TRACKER_FILE.is_file() else DEFAULT_TRACKER


@dataclass
class TrackResult:
    track_id: int
    bbox: tuple[int, int, int, int]
    label: str = "Белгісіз"
    confidence: float = 0.0
    status: str = "unknown"
    cadet_id: int | None = None


def _head_crop(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    """The face is near the top of a person's detection box; cropping just that region before
    running face detection is faster and avoids picking up a face from someone standing behind."""
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    y_end = y1 + max(int(h * 0.45), 40)
    x1, y1 = max(0, x1 - 10), max(0, y1 - 10)
    x2, y_end = min(frame.shape[1], x2 + 10), min(frame.shape[0], y_end + 10)
    if x2 - x1 < 20 or y_end - y1 < 20:
        return None
    return frame[y1:y_end, x1:x2]


class VisionPipeline:
    def __init__(self) -> None:
        self.model = None
        self.recognizer = RecognitionService()
        self.identity = TrackIdentityCache()
        self.last_inference_at = 0.0
        self.last_error: str | None = None

    def _load_detector(self):
        if self.model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("Ultralytics орнатылмаған (pip install ultralytics)") from exc
            self.model = YOLO(settings.yolo_model)
        return self.model

    def load_cadet_templates(self) -> None:
        self.recognizer.reload()

    def reset_tracking(self) -> None:
        self.identity.reset()

    def process(self, frame: np.ndarray) -> list[TrackResult]:
        now = monotonic()
        if now - self.last_inference_at < 1.0 / max(settings.processing_fps, 0.1):
            return self._current_results()
        self.last_inference_at = now

        try:
            model = self._load_detector()
            results = model.track(
                source=frame,
                persist=True,
                tracker=tracker_config(),
                verbose=False,
                conf=settings.yolo_conf,
                classes=[0],
                imgsz=settings.yolo_imgsz,
            )
            self.last_error = None
        except Exception as exc:
            # worker.status() surfaces this to the panel, so a failure here is visible
            # instead of silently producing an overlay without a single box.
            self.last_error = f"{type(exc).__name__}: {exc}"
            return self._current_results()

        if not results or results[0].boxes is None or results[0].boxes.id is None:
            self.identity.sweep(set())
            return []

        boxes = results[0].boxes
        ids = boxes.id.int().cpu().tolist()
        xyxy = boxes.xyxy.int().cpu().tolist()
        confs = boxes.conf.cpu().tolist()

        output: list[TrackResult] = []
        for track_id, box, conf in zip(ids, xyxy, confs):
            x1, y1, x2, y2 = box
            state = self.identity.touch(track_id, (x1, y1, x2, y2))
            self._maybe_recognize(frame, state)
            output.append(
                TrackResult(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    label=state.full_name or ("Белгісіз" if state.status != "ambiguous" else "Анықталмады"),
                    confidence=float(state.confidence or conf),
                    status=state.status,
                    cadet_id=state.cadet_id,
                )
            )
        self.identity.sweep(set(ids))
        return output

    def _maybe_recognize(self, frame: np.ndarray, state) -> None:
        if not self.identity.should_probe(state):
            return
        crop = _head_crop(frame, state.bbox)
        if crop is None or crop.size == 0:
            return
        try:
            face = face_engine.best_face(crop)
        except RuntimeError as exc:
            self.last_error = str(exc)
            return
        if face is None or face.quality < settings.recognition_min_quality:
            return  # no usable face this probe; keep whatever we already know and try again later

        if state.status != "recognized" and self.identity.try_relink(state, face.embedding):
            return

        match = self.recognizer.best_match(face.embedding)
        ambiguous = self.recognizer.is_ambiguous(match)
        confident = self.recognizer.is_confident(match)
        self.identity.register_vote(
            state,
            match.cadet_id if confident else None,
            match.full_name if confident else None,
            match.similarity,
            face.embedding,
            ambiguous,
        )

    def _current_results(self) -> list[TrackResult]:
        return [
            TrackResult(s.track_id, s.bbox, s.full_name or "Белгісіз", s.confidence, s.status, s.cadet_id)
            for s in self.identity.tracks.values()
        ]

    @staticmethod
    def draw(frame: np.ndarray, tracks: list[TrackResult]) -> np.ndarray:
        output = frame
        colors = {
            "recognized": (40, 190, 90),
            "candidate": (0, 165, 255),
            "ambiguous": (0, 90, 220),
            "unknown": (140, 140, 140),
        }
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            color = colors.get(track.status, (140, 140, 140))
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            text = track.label if track.status == "recognized" else f"{track.label} (T{track.track_id})"
            # draw_label keeps Kazakh/Cyrillic names readable (cv2.putText cannot).
            draw_label(output, text, (x1, max(24, y1 - 8)), color=color)
        return output
