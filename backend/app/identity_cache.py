"""Keeps a name attached to a tracker ID, so a recognized person is remembered while the
tracker keeps their ID (turned away, partly out of frame) - the algorithm does NOT re-run
face recognition every frame once someone is confirmed, only a person-detector + tracker.

It also handles the tracker losing someone entirely (fully hidden behind another person,
walked out and back) by keeping a short-lived gallery of "who was just confirmed, on which
now-possibly-dead track, with what embedding". A brand-new track ID is checked against this
gallery before it starts voting from scratch, so the same person gets their name back on
reappearance instead of being treated as a stranger.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from time import monotonic

import numpy as np

from .config import settings


@dataclass
class TrackState:
    track_id: int
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    status: str = "unknown"  # unknown | candidate | recognized | ambiguous
    cadet_id: int | None = None
    full_name: str | None = None
    confidence: float = 0.0
    votes: dict[int, int] = field(default_factory=dict)
    last_probe_at: float = 0.0
    confirmed_at: float = 0.0
    last_seen: float = field(default_factory=monotonic)


@dataclass
class _GalleryEntry:
    cadet_id: int
    full_name: str
    embedding: np.ndarray
    last_seen: float


class TrackIdentityCache:
    def __init__(self) -> None:
        self._lock = Lock()
        self.tracks: dict[int, TrackState] = {}
        self._gallery: dict[int, _GalleryEntry] = {}  # cadet_id -> last confirmed embedding

    def get(self, track_id: int) -> TrackState:
        with self._lock:
            state = self.tracks.get(track_id)
            if state is None:
                state = TrackState(track_id=track_id)
                self.tracks[track_id] = state
            return state

    def touch(self, track_id: int, bbox: tuple[int, int, int, int]) -> TrackState:
        state = self.get(track_id)
        state.bbox = bbox
        state.last_seen = monotonic()
        return state

    def should_probe(self, state: TrackState) -> bool:
        """Confirmed identities are re-checked rarely (just to catch a wrong early guess or a
        track-ID hijack); everything else is probed often so a name appears quickly."""
        interval = settings.reverify_seconds if state.status == "recognized" else settings.probe_seconds
        return monotonic() - state.last_probe_at >= interval

    def try_relink(self, state: TrackState, embedding: np.ndarray) -> bool:
        """Before voting from scratch on a new/unknown track, check the recent-identity
        gallery. Returns True and fills in the name if this is clearly someone who was just
        confirmed on a different (now probably-dead) track id."""
        now = monotonic()
        with self._lock:
            for cadet_id in [c for c, e in self._gallery.items() if now - e.last_seen > settings.relink_seconds]:
                del self._gallery[cadet_id]
            # "live elsewhere" means actually being updated by the tracker right now (last few
            # seconds), not merely still sitting in the map - a track that stopped getting
            # updates is exactly the "person got hidden" case this relink is meant to solve.
            freshness = max(3.0, settings.probe_seconds * 2)
            live_elsewhere = {
                s.cadet_id for s in self.tracks.values()
                if s.status == "recognized" and s.track_id != state.track_id and now - s.last_seen <= freshness
            }
            best_cadet, best_sim = None, 0.0
            for cadet_id, entry in self._gallery.items():
                if cadet_id in live_elsewhere:
                    continue  # that person is already confirmed on another live track
                sim = float(np.dot(embedding, entry.embedding))
                if sim > best_sim:
                    best_sim, best_cadet = sim, cadet_id
        if best_cadet is not None and best_sim >= settings.recognition_min_similarity + 0.05:
            entry = self._gallery[best_cadet]
            self._confirm(state, entry.cadet_id, entry.full_name, best_sim, embedding)
            return True
        return False

    def register_vote(
        self, state: TrackState, cadet_id: int | None, full_name: str | None,
        similarity: float, embedding: np.ndarray, ambiguous: bool,
    ) -> None:
        state.last_probe_at = monotonic()
        if ambiguous:
            state.status, state.votes = "ambiguous", {}
            return
        if cadet_id is None:
            if not state.votes:
                state.status = "unknown"
            return
        state.votes[cadet_id] = state.votes.get(cadet_id, 0) + 1
        top_cadet = max(state.votes, key=state.votes.get)
        if state.votes[top_cadet] >= settings.recognition_min_votes:
            self._confirm(state, top_cadet, full_name, similarity, embedding)
        else:
            state.status, state.cadet_id, state.full_name, state.confidence = (
                "candidate", top_cadet, full_name, similarity,
            )

    def _confirm(self, state: TrackState, cadet_id: int, full_name: str | None, similarity: float,
                 embedding: np.ndarray) -> None:
        state.status = "recognized"
        state.cadet_id, state.full_name, state.confidence = cadet_id, full_name, similarity
        state.confirmed_at = monotonic()
        state.votes = {}
        with self._lock:
            self._gallery[cadet_id] = _GalleryEntry(cadet_id, full_name or "", embedding, monotonic())

    def sweep(self, live_track_ids: set[int]) -> None:
        """Drop tracks the tracker no longer reports at all (gone for good, or TTL expired)."""
        now = monotonic()
        with self._lock:
            for track_id in [
                tid for tid, s in self.tracks.items()
                if tid not in live_track_ids and now - s.last_seen > settings.track_identity_ttl_seconds
            ]:
                del self.tracks[track_id]

    def reset(self) -> None:
        with self._lock:
            self.tracks.clear()
            self._gallery.clear()
