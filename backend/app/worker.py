from __future__ import annotations

import logging
import threading
import time
from typing import Iterator

import cv2

from .camera_sources import CameraSource, FrameGrabber, config_from_source
from .config import settings
from .pipeline import VisionPipeline

logger = logging.getLogger(__name__)


class CameraWorker:
    """Owns the capture thread and the per-frame vision pipeline."""

    def __init__(self) -> None:
        self.source = CameraSource(self._config())
        self.grabber: FrameGrabber | None = None
        self.pipeline = VisionPipeline()
        self.latest_jpeg: bytes | None = None
        self.latest_tracks: list[dict] = []
        self.running = False
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.error: str | None = None
        self._reported_pipeline_error: str | None = None

    @staticmethod
    def _config():
        return config_from_source(
            settings.camera_source,
            width=settings.camera_width,
            height=settings.camera_height,
            fps=settings.camera_fps,
            backend=settings.camera_backend,
        )

    def set_source(self, source: str, backend: str) -> None:
        was_running = self.running
        self.stop()
        settings.camera_source, settings.camera_backend = source, backend
        self.source = CameraSource(self._config())
        if was_running:
            self.start()

    def start(self) -> None:
        if self.running:
            return
        self.pipeline.load_cadet_templates()
        self.pipeline.reset_tracking()
        self.grabber = FrameGrabber(self.source)
        try:
            self.grabber.start()
        except Exception as exc:
            self.error = str(exc)
            self.grabber = None
            raise
        self.error = None
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True, name="camera-worker")
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.grabber:
            self.grabber.stop()
            self.grabber = None
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)

    def _note_pipeline_error(self) -> None:
        """Log a new pipeline failure once, so the console tells the same story as the panel."""
        error = self.pipeline.last_error
        if error and error != self._reported_pipeline_error:
            self._reported_pipeline_error = error
            logger.error("Vision pipeline error (кадрлар өңделмейді): %s", error)
        elif not error:
            self._reported_pipeline_error = None

    def _run(self) -> None:
        last_seq = -1
        target_width = settings.stream_max_width
        while self.running:
            seq, frame = self.grabber.latest()
            if frame is None or seq == last_seq:
                time.sleep(0.02)
                continue
            last_seq = seq
            self.error = self.grabber.error
            try:
                if frame.shape[1] > target_width:
                    scale = target_width / frame.shape[1]
                    frame = cv2.resize(frame, (target_width, int(frame.shape[0] * scale)))
                tracks = self.pipeline.process(frame)
                self._note_pipeline_error()
                rendered = self.pipeline.draw(frame, tracks)
                ok, encoded = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self.lock:
                        self.latest_jpeg = encoded.tobytes()
                        self.latest_tracks = [t.__dict__.copy() for t in tracks]
            except Exception as exc:
                self.error = str(exc)
                time.sleep(0.5)

    def mjpeg(self) -> Iterator[bytes]:
        while self.running:
            with self.lock:
                jpeg = self.latest_jpeg
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.05)

    def status(self) -> dict:
        with self.lock:
            tracks = list(self.latest_tracks)
        properties = self.source.actual_properties() if self.running else {}
        # Vision failures (tracker/recognition) belong in "error" as well: the panel prints
        # that field, and without it the user only saw "0 tracks" on a working picture.
        pipeline_error = self.pipeline.last_error
        return {
            "running": self.running,
            "error": self.error or pipeline_error,
            "pipeline_error": pipeline_error,
            "camera_fps": round(self.grabber.measured_fps, 1) if self.grabber else 0,
            "properties": properties,
            "tracks": tracks,
        }
