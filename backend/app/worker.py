from __future__ import annotations

import threading
import time
from typing import Iterator

import cv2

from .camera_sources import CameraSource, config_from_source
from .config import settings
from .database import list_cadets
from .pipeline import VisionPipeline


class CameraWorker:
    def __init__(self) -> None:
        self.source = CameraSource(
            config_from_source(
                settings.camera_source,
                width=settings.camera_width,
                height=settings.camera_height,
                fps=settings.camera_fps,
                backend=settings.camera_backend,
            )
        )
        self.pipeline = VisionPipeline()
        self.latest_jpeg: bytes | None = None
        self.latest_tracks: list[dict] = []
        self.running = False
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()

    def start(self) -> None:
        if self.running:
            return
        self.pipeline.load_cadet_templates(list_cadets())
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True, name="camera-worker")
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        self.source.release()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

    def _run(self) -> None:
        try:
            self.source.open()
        except Exception as exc:
            with self.lock:
                self.latest_tracks = [{"status": "error", "label": str(exc)}]
            self.running = False
            return
        while self.running:
            ok, frame = self.source.read()
            if not ok or frame is None:
                time.sleep(0.5)
                continue
            try:
                tracks = self.pipeline.process(frame)
                rendered = self.pipeline.draw(frame, tracks)
                success, encoded = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if success:
                    with self.lock:
                        self.latest_jpeg = encoded.tobytes()
                        self.latest_tracks = [track.__dict__.copy() for track in tracks]
            except Exception as exc:
                with self.lock:
                    self.latest_tracks = [{"status": "error", "label": str(exc)}]
                time.sleep(1)

    def mjpeg(self) -> Iterator[bytes]:
        while self.running:
            with self.lock:
                jpeg = self.latest_jpeg
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.05)

    def status(self) -> dict:
        with self.lock:
            return {
                "running": self.running,
                "properties": self.source.actual_properties(),
                "tracks": self.latest_tracks,
            }
