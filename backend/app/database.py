"""SQLite storage. Every query lives here so a later move to another database touches one file."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

from .config import settings


def _db_path():
    return settings.data_dir / "classroom_face.db"


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cadets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_code TEXT NOT NULL UNIQUE,
                full_name TEXT NOT NULL,
                group_name TEXT NOT NULL DEFAULT '',
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # One row per enrolled photo (a cadet can have several, which makes recognition more
        # robust). photo_file is a random name, never derived from user input.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS face_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cadet_id INTEGER NOT NULL REFERENCES cadets(id) ON DELETE CASCADE,
                photo_file TEXT NOT NULL,
                embedding BLOB NOT NULL,
                quality_score REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # One row per cadet marks them as already picked in the CURRENT random-pick cycle.
        # "Reset" just clears this table, so everyone becomes pickable again.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pick_history (
                cadet_id INTEGER PRIMARY KEY REFERENCES cadets(id) ON DELETE CASCADE,
                picked_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


_LIST_SQL = (
    "SELECT c.id, c.student_code, c.full_name, c.group_name, c.active, c.created_at, "
    "(SELECT COUNT(*) FROM face_templates t WHERE t.cadet_id = c.id) AS photo_count, "
    "(SELECT 1 FROM pick_history p WHERE p.cadet_id = c.id) AS picked "
    "FROM cadets c"
)


def _row(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["recognition_ready"] = data["photo_count"] > 0
    data["picked"] = bool(data.get("picked"))
    return data


def list_cadets(include_inactive: bool = False) -> list[dict[str, Any]]:
    init_db()
    where = "" if include_inactive else " WHERE c.active = 1"
    with _connect() as conn:
        rows = conn.execute(_LIST_SQL + where + " ORDER BY c.full_name").fetchall()
    return [_row(r) for r in rows]


def get_cadet(cadet_id: int) -> dict[str, Any] | None:
    init_db()
    with _connect() as conn:
        row = conn.execute(_LIST_SQL + " WHERE c.id = ?", (cadet_id,)).fetchone()
    return _row(row) if row else None


def create_cadet(student_code: str, full_name: str, group_name: str = "") -> dict[str, Any]:
    """A previously deleted (inactive) cadet with the same code is reactivated with a clean
    slate (old photos/embeddings dropped) rather than raising a duplicate error."""
    init_db()
    with _connect() as conn:
        existing = conn.execute(
            "SELECT id, active FROM cadets WHERE student_code = ?", (student_code,)
        ).fetchone()
        if existing and existing["active"]:
            raise sqlite3.IntegrityError("UNIQUE constraint failed: cadets.student_code")
        if existing:
            conn.execute("DELETE FROM face_templates WHERE cadet_id = ?", (existing["id"],))
            conn.execute(
                "UPDATE cadets SET full_name = ?, group_name = ?, active = 1 WHERE id = ?",
                (full_name, group_name, existing["id"]),
            )
            cadet_id = existing["id"]
        else:
            cursor = conn.execute(
                "INSERT INTO cadets (student_code, full_name, group_name) VALUES (?, ?, ?)",
                (student_code, full_name, group_name),
            )
            cadet_id = cursor.lastrowid
    return get_cadet(int(cadet_id))  # type: ignore[return-value]


def update_cadet(cadet_id: int, full_name: str | None, group_name: str | None) -> dict[str, Any] | None:
    init_db()
    with _connect() as conn:
        if full_name is not None:
            conn.execute("UPDATE cadets SET full_name = ? WHERE id = ?", (full_name, cadet_id))
        if group_name is not None:
            conn.execute("UPDATE cadets SET group_name = ? WHERE id = ?", (group_name, cadet_id))
    return get_cadet(cadet_id)


def delete_cadet(cadet_id: int) -> bool:
    """Soft delete: disappears from lists and recognition, but data stays on disk."""
    init_db()
    with _connect() as conn:
        cursor = conn.execute("UPDATE cadets SET active = 0 WHERE id = ? AND active = 1", (cadet_id,))
    return cursor.rowcount > 0


def add_template(cadet_id: int, photo_file: str, embedding: np.ndarray, quality: float) -> int:
    init_db()
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO face_templates (cadet_id, photo_file, embedding, quality_score) VALUES (?, ?, ?, ?)",
            (cadet_id, photo_file, np.asarray(embedding, dtype=np.float32).tobytes(), float(quality)),
        )
        return int(cursor.lastrowid)


def list_templates(cadet_id: int) -> list[dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, photo_file, quality_score, created_at FROM face_templates "
            "WHERE cadet_id = ? ORDER BY id",
            (cadet_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_template(cadet_id: int, template_id: int) -> str | None:
    """Returns the removed template's photo filename, or None if it did not exist."""
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT photo_file FROM face_templates WHERE id = ? AND cadet_id = ?",
            (template_id, cadet_id),
        ).fetchone()
        if not row:
            return None
        conn.execute("DELETE FROM face_templates WHERE id = ?", (template_id,))
    return row["photo_file"]


def load_active_templates() -> list[tuple[int, str, np.ndarray]]:
    """(cadet_id, full_name, unit-length embedding) for every template of every active cadet."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT c.id AS cadet_id, c.full_name, t.embedding FROM face_templates t "
            "JOIN cadets c ON c.id = t.cadet_id WHERE c.active = 1"
        ).fetchall()
    result = []
    for row in rows:
        vector = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        norm = np.linalg.norm(vector)
        if norm:
            result.append((int(row["cadet_id"]), row["full_name"], vector / norm))
    return result


# --- Random pick -----------------------------------------------------------------------

def pick_random_cadet() -> dict[str, Any] | None:
    """Pick a random active cadet who hasn't been picked yet this cycle, and mark them picked.
    Returns None once everyone active has been picked (call reset_picks to start a new cycle)."""
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT c.id, c.full_name, c.student_code, c.group_name FROM cadets c "
            "WHERE c.active = 1 AND c.id NOT IN (SELECT cadet_id FROM pick_history) "
            "ORDER BY RANDOM() LIMIT 1"
        ).fetchone()
        if not row:
            return None
        conn.execute("INSERT OR IGNORE INTO pick_history (cadet_id) VALUES (?)", (row["id"],))
    return dict(row)


def reset_picks() -> None:
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM pick_history")


def pick_progress() -> dict[str, int]:
    init_db()
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM cadets WHERE active = 1").fetchone()["n"]
        picked = conn.execute("SELECT COUNT(*) AS n FROM pick_history").fetchone()["n"]
    return {"total": total, "picked": picked, "remaining": max(0, total - picked)}
