import threading
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from .camera_sources import scan_devices
from .config import RESOURCE_ROOT, settings
from .database import (
    add_template,
    create_cadet,
    delete_cadet,
    delete_template,
    get_cadet,
    init_db,
    list_cadets,
    list_templates,
    pick_progress,
    pick_random_cadet,
    reset_picks,
    update_cadet,
)
from .enrollment import EnrollmentError, build_template
from .face_engine import engine as face_engine
from .settings_store import load_into_settings, save_camera_choice
from .worker import CameraWorker

app = FastAPI(title=settings.app_name, version="0.4.0")
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
    load_into_settings()
    init_db()
    # Loading InsightFace takes 5-60s; doing it in the background means the panel opens
    # immediately and the first "add cadet" / "start camera" click is fast, not one that hangs.
    threading.Thread(target=face_engine.warmup, daemon=True).start()


@app.on_event("shutdown")
def shutdown() -> None:
    worker.stop()


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(FRONTEND_INDEX)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "app": settings.app_name, "face_engine_ready": face_engine.loaded}


# --- Camera -----------------------------------------------------------------------------

@app.get("/api/camera/config")
def camera_config() -> dict:
    return {"source": settings.camera_source, "backend": settings.camera_backend}


@app.post("/api/camera/config")
def set_camera_config(source: str, backend: str = "auto") -> dict:
    if not source.strip():
        raise HTTPException(status_code=422, detail="Камера көзі бос болмауы керек")
    save_camera_choice(source.strip(), backend)
    worker.set_source(source.strip(), backend)
    return camera_config()


@app.get("/api/camera/devices")
def camera_devices() -> list[dict]:
    """USB/monoblock cameras found on this computer, for a dropdown instead of guessing a number."""
    busy = int(settings.camera_source) if settings.camera_source.isdigit() and worker.running else None
    return scan_devices(busy_index=busy)


@app.post("/api/camera/start")
def start_camera() -> dict:
    try:
        worker.start()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
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


# --- Cadets -----------------------------------------------------------------------------

@app.get("/api/cadets")
def cadets() -> list[dict]:
    return list_cadets()


@app.get("/api/cadets/{cadet_id}")
def cadet(cadet_id: int) -> dict:
    result = get_cadet(cadet_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Курсант табылмады")
    result["photos"] = list_templates(cadet_id)
    return result


@app.post("/api/cadets", status_code=201)
async def add_cadet(
    student_code: Annotated[str, Form()],
    full_name: Annotated[str, Form()],
    group_name: Annotated[str, Form()] = "",
    photo: UploadFile = File(...),
) -> dict:
    student_code, full_name = student_code.strip(), full_name.strip()
    if not student_code or not full_name:
        raise HTTPException(status_code=422, detail="Курсант ID және аты-жөні міндетті")
    if photo.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Тек JPG, PNG немесе WEBP сурет қабылданады")
    try:
        cadet_row = create_cadet(student_code, full_name, group_name.strip())
    except Exception as exc:
        if "UNIQUE constraint failed" in str(exc):
            raise HTTPException(status_code=409, detail="Бұл курсант ID бұрыннан бар") from exc
        raise
    try:
        filename, embedding, quality = build_template(await photo.read())
        add_template(cadet_row["id"], filename, embedding, quality)
    except EnrollmentError as exc:
        # Cadet profile is kept even if the first photo fails - the person can add one later.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        worker.pipeline.load_cadet_templates()
    return get_cadet(cadet_row["id"])  # type: ignore[return-value]


@app.patch("/api/cadets/{cadet_id}")
def edit_cadet(cadet_id: int, full_name: str | None = None, group_name: str | None = None) -> dict:
    result = update_cadet(cadet_id, full_name, group_name)
    if result is None:
        raise HTTPException(status_code=404, detail="Курсант табылмады")
    return result


@app.delete("/api/cadets/{cadet_id}")
def remove_cadet(cadet_id: int) -> dict[str, bool]:
    if not delete_cadet(cadet_id):
        raise HTTPException(status_code=404, detail="Курсант табылмады")
    worker.pipeline.load_cadet_templates()
    return {"deleted": True}


@app.post("/api/cadets/{cadet_id}/photos", status_code=201)
async def add_photo(cadet_id: int, photo: UploadFile = File(...)) -> dict:
    if get_cadet(cadet_id) is None:
        raise HTTPException(status_code=404, detail="Курсант табылмады")
    if photo.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Тек JPG, PNG немесе WEBP сурет қабылданады")
    try:
        filename, embedding, quality = build_template(await photo.read())
    except EnrollmentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    add_template(cadet_id, filename, embedding, quality)
    worker.pipeline.load_cadet_templates()
    return get_cadet(cadet_id)  # type: ignore[return-value]


@app.delete("/api/cadets/{cadet_id}/photos/{template_id}")
def remove_photo(cadet_id: int, template_id: int) -> dict[str, bool]:
    if delete_template(cadet_id, template_id) is None:
        raise HTTPException(status_code=404, detail="Фото табылмады")
    worker.pipeline.load_cadet_templates()
    return {"deleted": True}


# --- Random pick --------------------------------------------------------------------------

@app.post("/api/pick/random")
def pick_random() -> dict:
    result = pick_random_cadet()
    if result is None:
        raise HTTPException(
            status_code=409,
            detail="Барлық курсант таңдалды. Жаңа циклды бастау үшін \"Тізімді жаңарту\" басыңыз.",
        )
    return {**result, **pick_progress()}


@app.post("/api/pick/reset")
def pick_reset() -> dict:
    reset_picks()
    return pick_progress()


@app.get("/api/pick/status")
def pick_status() -> dict:
    return pick_progress()
