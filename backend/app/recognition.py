"""Matches a face embedding against every enrolled cadet's templates."""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import numpy as np

from .config import settings
from .database import load_active_templates


@dataclass
class Match:
    cadet_id: int | None
    full_name: str | None
    similarity: float
    margin: float  # gap to the second-best candidate; a small margin means "ambiguous"


class RecognitionService:
    def __init__(self) -> None:
        self._templates: list[tuple[int, str, np.ndarray]] = []
        self._lock = Lock()

    def reload(self) -> None:
        with self._lock:
            self._templates = load_active_templates()

    def best_match(self, embedding: np.ndarray) -> Match:
        with self._lock:
            templates = list(self._templates)
        if not templates:
            return Match(None, None, 0.0, 0.0)
        scored = sorted(
            ((cadet_id, name, float(np.dot(embedding, vector))) for cadet_id, name, vector in templates),
            key=lambda item: item[2],
            reverse=True,
        )
        best = scored[0]
        second = scored[1][2] if len(scored) > 1 else 0.0
        return Match(best[0], best[1], best[2], best[2] - second)

    def is_confident(self, match: Match) -> bool:
        return (
            match.cadet_id is not None
            and match.similarity >= settings.recognition_min_similarity
            and match.margin >= settings.recognition_min_margin
        )

    def is_ambiguous(self, match: Match) -> bool:
        """Two different cadets scored almost the same - better to say nothing than guess wrong."""
        return (
            match.cadet_id is not None
            and match.similarity >= settings.recognition_min_similarity
            and match.margin < settings.recognition_min_margin
        )
