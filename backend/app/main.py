from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database import create_cadet, delete_cadet, get_cadet, init_db, list_cadets
from .enrollment import EnrollmentError, create_embedding, save_photo

app = FastAPI(title=settings.app_name, version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()


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
        return {
            **result,
            "recognition_ready": embedding_path is not None,
            "message": "Курсант қосылды" if embedding_path else "Курсант қосылды; InsightFace кейін орнатылады",
        }
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
    return {"deleted": True}
