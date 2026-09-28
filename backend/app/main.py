from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from .camera_sources import CameraSource, config_from_source
from .config import RESOURCE_ROOT, settings
from .database import create_cadet, delete_cadet, get_cadet, init_db, list_cadets
from .enrollment import EnrollmentError, create_embedding, save_photo
from .worker import CameraWorker

app = FastAPI(title=settings.app_name, version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
worker = CameraWorker()
FRONTEND_INDEX = RESOURCE_ROOT / "frontend" / "index.html"


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.on_event("shutdown")
def shutdown() -> None:
    worker.stop()


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(FRONTEND_INDEX)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "app": settings.app_name, "version": "0.3.0"}


@app.get("/api/camera/config")
def camera_config() -> dict:
    return {
        "source": settings.camera_source,
        "backend": settings.camera_backend,
        "capture_resolution": f"{settings.camera_width}x{settings.camera_height}",
        "processing_resolution": f"{settings.processing_width}x{settings.processing_height}",
        "processing_fps": settings.processing_fps,
        "supported_sources": ["device index", "rtsp:// URL", "http:// URL", "video file"],
    }


@app.post("/api/camera/config")
def set_camera_config(source: str, backend: str = "auto") -> dict:
    if not source.strip():
        raise HTTPException(status_code=422, detail="Camera source бос болмауы керек")
    worker.stop()
    settings.camera_source = source.strip()
    settings.camera_backend = backend
    worker.source = CameraSource(
        config_from_source(
            settings.camera_source,
            width=settings.camera_width,
            height=settings.camera_height,
            fps=settings.camera_fps,
            backend=settings.camera_backend,
        )
    )
    return camera_config()


@app.post("/api/camera/start")
def start_camera() -> dict:
    worker.start()
    return {"started": True}


@app.post("/api/camera/stop")
def stop_camera() -> dict:
    worker.stop()
    return {"stopped": True}


@app.get("/api/camera/status")
def camera_status() -> dict:
    return worker.status()


@app.get("/api/camera/stream")
def camera_stream() -> StreamingResponse:
    if not worker.running:
        worker.start()
    return StreamingResponse(worker.mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/cadets")
def cadets() -> list[dict]:
    return list_cadets()


@app.get("/api/cadets/{cadet_id}")
def cadet(cadet_id: int) -> dict:
    result = get_cadet(cadet_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Курсант табылмады")
    return result


@app.post("/api/cadets", status_code=201)
async def add_cadet(
    student_code: Annotated[str, Form()],
    full_name: Annotated[str, Form()],
    group_name: Annotated[str, Form()] = "",
    photo: UploadFile = File(...),
) -> dict:
    student_code = student_code.strip()
    full_name = full_name.strip()
    if not student_code or not full_name:
        raise HTTPException(status_code=422, detail="student_code және full_name міндетті")
    if photo.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Тек JPG, PNG немесе WEBP сурет қабылданады")
    try:
        photo_path = save_photo(await photo.read(), student_code)
        embedding_path = create_embedding(photo_path, student_code)
        result = create_cadet(
            student_code,
            full_name,
            group_name.strip(),
            str(photo_path),
            str(embedding_path) if embedding_path else None,
        )
        worker.pipeline.load_cadet_templates(list_cadets())
        return {**result, "recognition_ready": embedding_path is not None}
    except EnrollmentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        if "UNIQUE constraint failed" in str(exc):
            raise HTTPException(status_code=409, detail="Бұл student_code бұрыннан бар") from exc
        raise


@app.delete("/api/cadets/{cadet_id}")
def remove_cadet(cadet_id: int) -> dict[str, bool]:
    if not delete_cadet(cadet_id):
        raise HTTPException(status_code=404, detail="Курсант табылмады")
    worker.pipeline.load_cadet_templates(list_cadets())
    return {"deleted": True}
