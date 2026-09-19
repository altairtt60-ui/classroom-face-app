import sqlite3
from pathlib import Path
from typing import Any

from .config import settings

DB_PATH = settings.data_dir / "classroom_face.db"


def init_db() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cadets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_code TEXT NOT NULL UNIQUE,
                full_name TEXT NOT NULL,
                group_name TEXT NOT NULL DEFAULT '',
                photo_path TEXT,
                embedding_path TEXT,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def list_cadets() -> list[dict[str, Any]]:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, student_code, full_name, group_name, photo_path, active, created_at "
            "FROM cadets WHERE active = 1 ORDER BY full_name"
        ).fetchall()
    return [dict(row) for row in rows]


def get_cadet(cadet_id: int) -> dict[str, Any] | None:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT id, student_code, full_name, group_name, photo_path, embedding_path, active, created_at "
            "FROM cadets WHERE id = ?",
            (cadet_id,),
        ).fetchone()
    return dict(row) if row else None


def create_cadet(
    student_code: str,
    full_name: str,
    group_name: str,
    photo_path: str,
    embedding_path: str | None,
) -> dict[str, Any]:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "INSERT INTO cadets (student_code, full_name, group_name, photo_path, embedding_path) "
            "VALUES (?, ?, ?, ?, ?)",
            (student_code, full_name, group_name, photo_path, embedding_path),
        )
        conn.commit()
        cadet_id = cursor.lastrowid
    return get_cadet(int(cadet_id))  # type: ignore[return-value]


def delete_cadet(cadet_id: int) -> bool:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("UPDATE cadets SET active = 0 WHERE id = ?", (cadet_id,))
        conn.commit()
    return cursor.rowcount > 0
