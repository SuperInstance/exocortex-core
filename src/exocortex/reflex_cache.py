""".nail reflex cache — sub-millisecond pattern-response lookup."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ._embed import hash_embedding


@dataclass
class NailReflex:
    """A compiled reflex record stored as a ``.nail`` JSON file."""

    id: str
    situation: str
    match_key: str
    response: str
    confidence: float
    source: str
    max_consecutive_uses: int
    embedding: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("reflex id is required")
        self.confidence = float(self.confidence)


EmbedFn = Callable[[str], List[float]]


def _normalize(text: str) -> str:
    """Collapse whitespace and lower-case for stable hashing."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _sig_hash(signature: str) -> str:
    """Return a 16-char hex hash of the signature for O(1) lookup."""
    return hashlib.sha256(_normalize(signature).encode("utf-8")).hexdigest()[:16]


class ReflexCache:
    """In-memory reflex cache with exact-hash and vector-nearest-neighbour lookup.

    Each reflex carries an asymmetric confidence update and a
    ``max_consecutive_uses`` escape hatch so high-confidence wrong answers
    cannot become permanent blind spots.
    """

    DEFAULT_THRESHOLD = 0.85
    DEFAULT_MAX_CONSECUTIVE = 50

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_THRESHOLD,
        embed_fn: Optional[EmbedFn] = None,
        max_consecutive_uses_default: int = DEFAULT_MAX_CONSECUTIVE,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.embed_fn = embed_fn or hash_embedding
        self.max_consecutive_uses_default = max_consecutive_uses_default

        # sig_hash -> reflex id
        self._exact: Dict[str, str] = {}
        # reflex id -> NailReflex
        self._by_id: Dict[str, NailReflex] = {}
        # reflex id -> consecutive uses since last reset
        self._consecutive: Dict[str, int] = {}

        # Caching for vector nearest-neighbour search.
        self._matrix_stale = True
        self._embedding_matrix: Optional[np.ndarray] = None
        self._matrix_ids: List[str] = []

    # ------------------------------------------------------------------
    # Core lookup
    # ------------------------------------------------------------------

    def lookup(
        self,
        signature: str,
        embedding: Optional[List[float]] = None,
    ) -> Optional[NailReflex]:
        """Return a matching reflex, or ``None`` if none passes the gate."""
        # 1) Exact hash lookup
        sig_hash = _sig_hash(signature)
        reflex_id = self._exact.get(sig_hash)
        if reflex_id is not None:
            reflex = self._by_id.get(reflex_id)
            if reflex is not None:
                if self._can_use(reflex_id, reflex):
                    self._record_hit(reflex_id)
                    return reflex
                # Exact match exists but is blocked by the escape hatch;
                # do not fall back to the same reflex via vector search.
                return None

        # 2) Vector nearest-neighbour fallback
        if embedding is None:
            embedding = self.embed_fn(signature)
        nearest = self._nearest(embedding)
        if nearest is not None:
            rid, reflex = nearest
            if self._can_use(rid, reflex):
                self._record_hit(rid)
                return reflex

        return None

    def _can_use(self, reflex_id: str, reflex: NailReflex) -> bool:
        if reflex.confidence < self.confidence_threshold:
            return False
        # Escape-hatch: force a re-check after N consecutive identical dispatches.
        if self._consecutive.get(reflex_id, 0) >= reflex.max_consecutive_uses:
            self._consecutive[reflex_id] = 0
            return False
        return True

    def _record_hit(self, reflex_id: str) -> None:
        self._consecutive[reflex_id] = self._consecutive.get(reflex_id, 0) + 1

    def _nearest(
        self,
        embedding: List[float],
    ) -> Optional[tuple[str, NailReflex]]:
        """Cosine nearest neighbour above the confidence threshold."""
        if not self._by_id:
            return None

        if self._matrix_stale:
            ids: List[str] = []
            rows: List[np.ndarray] = []
            for rid, reflex in self._by_id.items():
                if reflex.confidence >= self.confidence_threshold and reflex.embedding:
                    ids.append(rid)
                    rows.append(np.asarray(reflex.embedding, dtype=float))
            if rows:
                self._embedding_matrix = np.stack(rows)
                self._matrix_ids = ids
            else:
                self._embedding_matrix = None
                self._matrix_ids = []
            self._matrix_stale = False

        if self._embedding_matrix is None or len(self._matrix_ids) == 0:
            return None

        q = np.asarray(embedding, dtype=float)
        dots = self._embedding_matrix @ q
        norms = np.linalg.norm(self._embedding_matrix, axis=1) * np.linalg.norm(q)
        with np.errstate(divide="ignore", invalid="ignore"):
            sims = np.where(norms > 0, dots / norms, 0.0)

        idx = int(np.argmax(sims))
        best_score = float(sims[idx])
        if best_score >= 0.85:
            rid = self._matrix_ids[idx]
            return rid, self._by_id[rid]
        return None

    # ------------------------------------------------------------------
    # Storage / mutation
    # ------------------------------------------------------------------

    def store(
        self,
        signature: str,
        response: str,
        confidence: float,
        source: str = "local",
        situation: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        max_consecutive_uses: Optional[int] = None,
        reflex_id: Optional[str] = None,
    ) -> str:
        """Compile a new reflex and index it by exact hash and vector."""
        rid = reflex_id or secrets.token_hex(8)
        if max_consecutive_uses is None:
            max_consecutive_uses = self.max_consecutive_uses_default

        normalized = _normalize(signature)
        reflex = NailReflex(
            id=rid,
            situation=situation or normalized,
            match_key=normalized,
            response=response,
            confidence=max(0.05, min(0.95, float(confidence))),
            source=source,
            max_consecutive_uses=max_consecutive_uses,
            embedding=self.embed_fn(normalized),
            metadata=dict(metadata or {}),
        )

        self._by_id[rid] = reflex
        self._exact[_sig_hash(signature)] = rid
        self._consecutive[rid] = 0
        self._matrix_stale = True
        return rid

    def update_confidence(self, reflex_id: str, success: bool) -> Optional[float]:
        """Apply Pincher's asymmetric confidence rule."""
        reflex = self._by_id.get(reflex_id)
        if reflex is None:
            return None

        c = reflex.confidence
        if success:
            c = c + 0.05 * (1.0 - c)
        else:
            c = c - 0.10 * c
        c = max(0.05, min(0.95, c))
        reflex.confidence = round(c, 6)
        self._matrix_stale = True
        return reflex.confidence

    def reset_consecutive(self, reflex_id: str) -> None:
        """Reset the escape-hatch counter (used after a re-verification)."""
        if reflex_id in self._consecutive:
            self._consecutive[reflex_id] = 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write ``.nail`` records to a directory or a single JSON file."""
        target = Path(path)
        if target.suffix == ".json" or not target.exists() and str(target).endswith(".json"):
            target.parent.mkdir(parents=True, exist_ok=True)
            records = [asdict(r) for r in self._by_id.values()]
            target.write_text(json.dumps(records, indent=2), encoding="utf-8")
        else:
            target.mkdir(parents=True, exist_ok=True)
            for reflex in self._by_id.values():
                nail_path = target / f"{reflex.id}.nail"
                nail_path.write_text(
                    json.dumps(asdict(reflex), indent=2), encoding="utf-8"
                )

    def load(self, path: str | Path) -> int:
        """Load ``.nail`` records from a directory or a single JSON file."""
        target = Path(path)
        if target.is_file():
            text = target.read_text(encoding="utf-8")
            records = json.loads(text)
        elif target.is_dir():
            records = []
            for nail_path in target.glob("*.nail"):
                records.append(json.loads(nail_path.read_text(encoding="utf-8")))
        else:
            raise FileNotFoundError(f"No such nail file or directory: {target}")

        loaded = 0
        for rec in records:
            reflex = NailReflex(**rec)
            self._by_id[reflex.id] = reflex
            self._exact[_sig_hash(reflex.match_key)] = reflex.id
            self._consecutive.setdefault(reflex.id, 0)
            loaded += 1
        self._matrix_stale = True
        return loaded

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def size(self) -> int:
        return len(self._by_id)

    def stats(self) -> Dict[str, Any]:
        total = len(self._by_id)
        if total == 0:
            return {"total": 0, "active": 0, "avg_confidence": 0.0}
        active = sum(
            1 for r in self._by_id.values() if r.confidence >= self.confidence_threshold
        )
        avg_conf = sum(r.confidence for r in self._by_id.values()) / total
        return {
            "total": total,
            "active": active,
            "avg_confidence": round(avg_conf, 4),
        }


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity of two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a**0.5 * norm_b**0.5)
