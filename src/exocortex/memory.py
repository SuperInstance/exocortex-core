"""Vectorize-backed semantic memory with pluggable backends."""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ._embed import hash_embedding


@dataclass
class MemoryEntry:
    """A single semantic-memory record."""

    id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class VectorBackend(ABC):
    """Abstract port for vector storage."""

    @abstractmethod
    def upsert(self, entry: MemoryEntry) -> None:
        ...

    @abstractmethod
    def query(
        self, embedding: List[float], k: int = 5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Return ``(doc_id, score, metadata)`` tuples, best first."""
        ...

    @abstractmethod
    def delete(self, doc_id: str) -> bool:
        ...

    @abstractmethod
    def count(self) -> int:
        ...


class InMemoryBackend(VectorBackend):
    """Linear-scan backend with no external dependencies."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self._entries: Dict[str, MemoryEntry] = {}

    def upsert(self, entry: MemoryEntry) -> None:
        if len(entry.embedding) != self.dim:
            raise ValueError(f"expected embedding dimension {self.dim}")
        self._entries[entry.id] = entry

    def query(
        self, embedding: List[float], k: int = 5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        if len(embedding) != self.dim:
            raise ValueError(f"expected embedding dimension {self.dim}")
        scored: List[Tuple[str, float]] = []
        for doc_id, entry in self._entries.items():
            score = _cosine_similarity(embedding, entry.embedding)
            scored.append((doc_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            (doc_id, score, dict(self._entries[doc_id].metadata))
            for doc_id, score in scored[:k]
        ]

    def delete(self, doc_id: str) -> bool:
        if doc_id in self._entries:
            del self._entries[doc_id]
            return True
        return False

    def count(self) -> int:
        return len(self._entries)


class SQLiteVecBackend(VectorBackend):
    """sqlite-vec backend for persistent local semantic memory.

    Falls back to raising :class:`ImportError` at construction time if
    ``sqlite_vec`` is not installed; :class:`MemoryIndex` catches this
    and switches to :class:`InMemoryBackend`.
    """

    def __init__(self, path: str | Path = ":memory:", dim: int = 384) -> None:
        try:
            import sqlite_vec  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("sqlite_vec is not installed") from exc

        self._dim = dim
        self._sqlite_vec = sqlite_vec
        self._db = sqlite3.connect(str(path))
        self._db.enable_load_extension(True)
        sqlite_vec.load(self._db)

        self._db.execute(
            "CREATE TABLE IF NOT EXISTS docs ("
            "rowid INTEGER PRIMARY KEY AUTOINCREMENT, "
            "doc_id TEXT UNIQUE NOT NULL, "
            "text TEXT NOT NULL, "
            "metadata TEXT NOT NULL DEFAULT '{}'"
            ")"
        )
        self._db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(embedding float[{dim}])"
        )
        self._db.commit()

    def upsert(self, entry: MemoryEntry) -> None:
        if len(entry.embedding) != self._dim:
            raise ValueError(f"expected embedding dimension {self._dim}")

        # Remove any existing row so the rowid stays in sync.
        self.delete(entry.id)

        cur = self._db.execute(
            "INSERT INTO docs (doc_id, text, metadata) VALUES (?, ?, ?)",
            (entry.id, entry.text, json.dumps(entry.metadata)),
        )
        rowid = cur.lastrowid
        self._db.execute(
            "INSERT INTO vec_items (rowid, embedding) VALUES (?, ?)",
            (rowid, json.dumps(entry.embedding)),
        )
        self._db.commit()

    def query(
        self, embedding: List[float], k: int = 5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        if len(embedding) != self._dim:
            raise ValueError(f"expected embedding dimension {self._dim}")

        results: List[Tuple[str, float, Dict[str, Any]]] = []
        rows = self._db.execute(
            "SELECT docs.doc_id, docs.metadata, vec_items.distance "
            "FROM vec_items "
            "JOIN docs ON docs.rowid = vec_items.rowid "
            "WHERE vec_items.embedding MATCH ? AND k = ? "
            "ORDER BY vec_items.distance",
            (json.dumps(embedding), k),
        ).fetchall()

        for doc_id, metadata_json, distance in rows:
            # sqlite-vec returns L2 distance; convert to a similarity-ish score.
            score = 1.0 / (1.0 + float(distance))
            results.append((doc_id, score, json.loads(metadata_json)))
        return results

    def delete(self, doc_id: str) -> bool:
        row = self._db.execute(
            "SELECT rowid FROM docs WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            return False
        rowid = row[0]
        self._db.execute("DELETE FROM vec_items WHERE rowid = ?", (rowid,))
        self._db.execute("DELETE FROM docs WHERE rowid = ?", (rowid,))
        self._db.commit()
        return True

    def count(self) -> int:
        row = self._db.execute("SELECT COUNT(*) FROM docs").fetchone()
        return row[0] if row else 0


# ---------------------------------------------------------------------------
# Public index
# ---------------------------------------------------------------------------

EmbedFn = Callable[[str], List[float]]


class MemoryIndex:
    """Semantic-memory index with pluggable vector backend.

    The default embedding is the deterministic hash embedding, and the default
    backend is sqlite-vec when available; otherwise it degrades gracefully to
    an in-memory linear scan.
    """

    DEFAULT_DIM = 384

    def __init__(
        self,
        backend: Optional[VectorBackend] = None,
        embed_fn: Optional[EmbedFn] = None,
        sqlite_path: Optional[str | Path] = None,
        dim: int = DEFAULT_DIM,
    ) -> None:
        self.embed_fn = embed_fn or hash_embedding
        self.dim = dim

        if backend is not None:
            self.backend = backend
        else:
            self.backend = self._default_backend(sqlite_path, dim)

    @staticmethod
    def _default_backend(
        sqlite_path: Optional[str | Path], dim: int
    ) -> VectorBackend:
        path = sqlite_path or ":memory:"
        try:
            return SQLiteVecBackend(path=path, dim=dim)
        except ImportError:
            return InMemoryBackend(dim=dim)

    def upsert(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> MemoryEntry:
        """Index a document. If no embedding is supplied, one is computed."""
        emb = embedding if embedding is not None else self.embed_fn(text)
        entry = MemoryEntry(
            id=doc_id,
            text=text,
            embedding=emb,
            metadata=dict(metadata or {}),
        )
        self.backend.upsert(entry)
        return entry

    def query(
        self,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        k: int = 5,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Return nearest neighbours for a text string or a raw embedding."""
        if embedding is None:
            if text is None:
                raise ValueError("Provide either text or embedding")
            embedding = self.embed_fn(text)
        return self.backend.query(embedding, k=k)

    def delete(self, doc_id: str) -> bool:
        return self.backend.delete(doc_id)

    def write_outcome(
        self,
        doc_id: str,
        outcome: Dict[str, Any],
    ) -> Optional[MemoryEntry]:
        """Append outcome metadata to an existing memory entry.

        This is the Craftmind write-back loop: every executed action writes
        its outcome back to the index.
        """
        # In-memory backend keeps the entry object; sqlite backend must re-read.
        if isinstance(self.backend, InMemoryBackend):
            entry = self.backend._entries.get(doc_id)
            if entry is None:
                return None
            entry.metadata.setdefault("outcomes", []).append(outcome)
            return entry

        if isinstance(self.backend, SQLiteVecBackend):
            row = self.backend._db.execute(
                "SELECT text, metadata FROM docs WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if row is None:
                return None
            text, metadata_json = row
            metadata = json.loads(metadata_json)
            metadata.setdefault("outcomes", []).append(outcome)
            self.backend._db.execute(
                "UPDATE docs SET metadata = ? WHERE doc_id = ?",
                (json.dumps(metadata), doc_id),
            )
            self.backend._db.commit()
            return MemoryEntry(id=doc_id, text=text, embedding=[], metadata=metadata)

        return None

    def count(self) -> int:
        return self.backend.count()


def _cosine_similarity(a: List[float], b: List[float]) -> float:
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
