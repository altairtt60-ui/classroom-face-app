"""Camera-independent frame source: USB/monoblock cameras, IP cameras (RTSP/HTTP), and video
files, all behind the same read()/reconnect()/release() interface so the rest of the app never
needs to know which kind is in use.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Iterator
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

# TCP is far more reliable than UDP for RTSP over Wi-Fi/LAN; must be set before the first open.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

FALLBACK_SIZES = [(1920, 1080), (1280, 720), (640, 480)]


class CameraKind(str, Enum):
    DEVICE = "device"
    RTSP = "rtsp"  # rtsp:// or http(s):// stream (IP camera, phone camera apps)
    VIDEO_FILE = "video_file"


@dataclass
class CameraConfig:
    kind: CameraKind = CameraKind.DEVICE
    source: str = "0"
    width: int = 1920
    height: int = 1080
    fps: int = 15
    backend: str = "auto"


def mask_source(source: str) -> str:
    """Hide the password of rtsp://user:pass@host URLs before showing them anywhere."""
    try:
        parts = urlsplit(source)
    except ValueError:
        return source
    if not parts.password:
        return source
    host = parts.hostname or ""
    if parts.port:
        host += f":{parts.port}"
    return urlunsplit((parts.scheme, f"{parts.username}:***@{host}", parts.path, parts.query, ""))


def _backend_order(backend: str) -> list[int]:
    if backend == "dshow":
        return [cv2.CAP_DSHOW]
    if backend == "msmf":
        return [cv2.CAP_MSMF]
    if sys.platform == "win32":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def _is_black(frame: np.ndarray | None) -> bool:
    return frame is None or frame.size == 0 or float(frame.mean()) < 1.0


class CameraSource:
    def __init__(self, config: CameraConfig):
        self.config = config
        self.capture: cv2.VideoCapture | None = None
        self.lock = Lock()
        self.notes: str = ""

    def _source_value(self) -> int | str:
        if self.config.kind == CameraKind.DEVICE:
            try:
                return int(self.config.source)
            except ValueError as exc:
                raise ValueError("Камера индексі бүтін сан болуы керек") from exc
        return self.config.source

    def _open_device(self, index: int) -> cv2.VideoCapture:
        """Try several backends/resolutions; USB cameras often return black frames at a
        resolution they cannot actually deliver, so we fall back instead of failing outright."""
        sizes = [(self.config.width, self.config.height)] + [
            s for s in FALLBACK_SIZES if s[0] < self.config.width
        ]
        for backend in _backend_order(self.config.backend):
            for width, height in sizes:
                capture = cv2.VideoCapture(index, backend)
                if not capture.isOpened():
                    capture.release()
                    break  # this backend cannot open the device at all, try the next backend
                capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                capture.set(cv2.CAP_PROP_FPS, self.config.fps)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                good = False
                for _ in range(20):  # auto-exposure needs a moment; early frames are often black
                    ok, frame = capture.read()
                    if ok and not _is_black(frame):
                        good = True
                        break
                    time.sleep(0.05)
                if good:
                    if (width, height) != (self.config.width, self.config.height):
                        self.notes = f"{self.config.width}x{self.config.height} қолдамайды, {width}x{height} қолданылды"
                    return capture
                capture.release()
        raise RuntimeError(
            f"Камера {index} ашылмады немесе тек қара кадр береді. Windows-та Параметры → "
            "Конфиденциальность → Камера рұқсатын және басқа бағдарлама (Zoom, Teams, браузер) "
            "камераны қолданбай тұрғанын тексеріңіз."
        )

    def open(self) -> None:
        self.notes = ""
        source = self._source_value()
        if self.config.kind == CameraKind.DEVICE:
            capture = self._open_device(int(source))
        else:
            capture = cv2.VideoCapture(source, cv2.CAP_ANY)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"Камера көзі ашылмады: {mask_source(self.config.source)}")
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
            if self.config.kind == CameraKind.VIDEO_FILE:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop a test video instead of stopping
                return False, None
            self.reconnect()
            return False, None
        return True, frame

    def reconnect(self) -> None:
        self.release()
        self.open()

    def actual_properties(self) -> dict:
        with self.lock:
            capture = self.capture
        if capture is None:
            return {}
        return {
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "note": self.notes,
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


class FrameGrabber:
    """Reads the camera in its own thread and keeps only the newest frame.

    Without this, a camera thread that also runs AI inference reads one frame, spends a
    second or more processing it, then reads again - so the picture falls further and
    further behind (this is what made a frame arrive every few minutes). With this, capture
    and processing run independently: the AI loop always grabs whatever is freshest.
    """

    def __init__(self, source: CameraSource):
        self.source = source
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._lock = Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        self.measured_fps = 0.0

    def start(self) -> None:
        self.source.open()  # raise here so the caller sees the real reason immediately
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="frame-grabber")
        self._thread.start()

    def _run(self) -> None:
        count, started = 0, time.monotonic()
        while not self._stop.is_set():
            try:
                ok, frame = self.source.read()
            except Exception as exc:  # camera unplugged, stream dropped, etc.
                self.error = str(exc)
                time.sleep(1.0)
                continue
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            self.error = None
            with self._lock:
                self._frame, self._seq = frame, self._seq + 1
            count += 1
            elapsed = time.monotonic() - started
            if elapsed >= 2.0:
                self.measured_fps, count, started = count / elapsed, 0, time.monotonic()

    def latest(self) -> tuple[int, np.ndarray | None]:
        with self._lock:
            return self._seq, self._frame

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self.source.release()


def config_from_source(source: str, **kwargs) -> CameraConfig:
    source = source.strip()
    if source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://")):
        kind = CameraKind.RTSP
    elif source.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        kind = CameraKind.VIDEO_FILE
    else:
        kind = CameraKind.DEVICE
    return CameraConfig(kind=kind, source=source, **kwargs)


def scan_devices(max_index: int = 6, busy_index: int | None = None) -> list[dict]:
    """Probe device indices 0..max_index for a connected USB/monoblock camera, so the UI can
    offer a dropdown instead of asking the person to guess a number."""
    found = []
    for index in range(max_index):
        if busy_index is not None and index == busy_index:
            found.append({"index": index, "label": f"Камера {index} (қазір қолданылуда)", "busy": True})
            continue
        opened = False
        for backend in _backend_order("auto"):
            capture = cv2.VideoCapture(index, backend)
            if capture.isOpened():
                ok, _frame = capture.read()
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                capture.release()
                if ok:
                    found.append({"index": index, "label": f"Камера {index} ({width}x{height})", "busy": False})
                    opened = True
                    break
            capture.release()
        if not opened:
            continue
    return found
