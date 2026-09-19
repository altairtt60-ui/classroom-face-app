from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Iterator

import cv2


class CameraKind(str, Enum):
    DEVICE = "device"
    RTSP = "rtsp"
    VIDEO_FILE = "video_file"


@dataclass
class CameraConfig:
    kind: CameraKind = CameraKind.DEVICE
    source: str = "0"
    width: int = 2560
    height: int = 1440
    fps: int = 15
    backend: str = "auto"


class CameraSource:
    """Camera-independent frame source.

    The rest of the application only depends on read/reconnect/release.
    """

    def __init__(self, config: CameraConfig):
        self.config = config
        self.capture: cv2.VideoCapture | None = None
        self.lock = Lock()

    def _source_value(self) -> int | str:
        if self.config.kind == CameraKind.DEVICE:
            try:
                return int(self.config.source)
            except ValueError as exc:
                raise ValueError("Device camera source must be an integer index") from exc
        return self.config.source

    def open(self) -> None:
        source = self._source_value()
        if self.config.backend == "dshow" and self.config.kind == CameraKind.DEVICE:
            backend = cv2.CAP_DSHOW
        elif self.config.backend == "msmf" and self.config.kind == CameraKind.DEVICE:
            backend = cv2.CAP_MSMF
        else:
            backend = cv2.CAP_ANY

        capture = cv2.VideoCapture(source, backend)
        if self.config.kind == CameraKind.DEVICE:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Camera source could not be opened: {self.config.source}")
        with self.lock:
            self.capture = capture

    def read(self):
        with self.lock:
            capture = self.capture
        if capture is None or not capture.isOpened():
            self.open()
            with self.lock:
                capture = self.capture
        ok, frame = capture.read()
        if not ok:
            self.reconnect()
            return False, None
        return True, frame

    def reconnect(self) -> None:
        self.release()
        self.open()

    def actual_properties(self) -> dict[str, float | int]:
        with self.lock:
            capture = self.capture
        if capture is None:
            return {}
        return {
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        }

    def frames(self) -> Iterator:
        while True:
            ok, frame = self.read()
            if ok:
                yield frame

    def release(self) -> None:
        with self.lock:
            if self.capture is not None:
                self.capture.release()
                self.capture = None


def config_from_source(source: str, **kwargs) -> CameraConfig:
    if source.startswith("rtsp://") or source.startswith("http://") or source.startswith("https://"):
        kind = CameraKind.RTSP
    elif source.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        kind = CameraKind.VIDEO_FILE
    else:
        kind = CameraKind.DEVICE
    return CameraConfig(kind=kind, source=source, **kwargs)
