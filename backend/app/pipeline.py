from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import cv2
import numpy as np

from .config import settings
from .recognition import RecognitionService


@dataclass
class TrackResult:
    track_id: int
    bbox: tuple[int, int, int, int]
    label: str = "Unknown"
    confidence: float = 0.0
    status: str = "unknown"


class VisionPipeline:
    """Lazy-loaded single-camera pipeline.

    Ultralytics owns detection/persistence for the first implementation. The camera source
    remains independent, so USB, RTSP and video-file sources use the same pipeline.
    """

    def __init__(self) -> None:
        self.model = None
        self.recognizer = RecognitionService()
        self.track_states: dict[int, TrackResult] = {}
        self.last_inference_at = 0.0

    def _load_detector(self):
        if self.model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("Ultralytics орнатылмаған. pip install ultralytics") from exc
            self.model = YOLO("yolo11n.pt")
        return self.model

    def load_cadet_templates(self, cadets: list[dict]) -> None:
        self.recognizer.load_templates(cadets)

    def process(self, frame: np.ndarray) -> list[TrackResult]:
        now = monotonic()
        if now - self.last_inference_at < 1.0 / max(settings.processing_fps, 1):
            return list(self.track_states.values())
        self.last_inference_at = now

        model = self._load_detector()
        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            conf=0.35,
            classes=[0],
            imgsz=640,
        )
        if not results:
            return []
        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return []

        output: list[TrackResult] = []
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else list(range(len(boxes)))
        xyxy = boxes.xyxy.int().cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        for index, box in enumerate(xyxy):
            x1, y1, x2, y2 = box
            track_id = int(ids[index])
            previous = self.track_states.get(track_id)
            current = previous or TrackResult(track_id, (x1, y1, x2, y2))
            current.bbox = (x1, y1, x2, y2)
            current.confidence = float(confs[index])
            output.append(current)
            self.track_states[track_id] = current
        return output

    @staticmethod
    def draw(frame: np.ndarray, tracks: list[TrackResult]) -> np.ndarray:
        output = frame.copy()
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            color = (40, 190, 90) if track.status == "recognized" else (0, 165, 255)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            text = f"{track.label} | T{track.track_id} | {track.confidence:.2f}"
            cv2.putText(output, text, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return output
