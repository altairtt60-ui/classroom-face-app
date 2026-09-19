from fastapi import FastAPI
from .config import settings

app = FastAPI(title=settings.app_name, version="0.1.0")


@app.get("/api/health")
def health() -> dict[str, str | int]:
    return {
        "status": "ok",
        "app": settings.app_name,
        "camera_index": settings.camera_index,
        "capture_resolution": f"{settings.camera_width}x{settings.camera_height}",
        "processing_resolution": f"{settings.processing_width}x{settings.processing_height}",
    }


@app.get("/api/camera/config")
def camera_config() -> dict[str, int]:
    return {
        "camera_index": settings.camera_index,
        "capture_width": settings.camera_width,
        "capture_height": settings.camera_height,
        "processing_width": settings.processing_width,
        "processing_height": settings.processing_height,
        "processing_fps": settings.processing_fps,
    }
