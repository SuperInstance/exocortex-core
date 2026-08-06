"""
Tests for exocortex.memory — MemoryEntry, InMemoryBackend, SQLiteVecBackend,
MemoryIndex, write_outcome, cosine similarity edge cases.

Targets the 68% coverage gap on memory.py.
"""
from __future__ import annotations

import json
import math
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from exocortex.memory import (
    MemoryEntry,
    VectorBackend,
    InMemoryBackend,
    SQLiteVecBackend,
    MemoryIndex,
    _cosine_similarity,
)


# ============================================================
#  MemoryEntry
# ============================================================

class TestMemoryEntry:
    def test_basic_creation(self):
        entry = MemoryEntry(id="a", text="hello", embedding=[1.0, 0.0])
        assert entry.id == "a"
        assert entry.text == "hello"
        assert entry.embedding == [1.0, 0.0]

    def test_default_metadata_is_empty_dict(self):
        entry = MemoryEntry(id="b", text="hi", embedding=[1.0])
        assert entry.metadata == {}

    def test_metadata_is_not_shared_between_instances(self):
        """Default factory should create new dict per instance."""
        e1 = MemoryEntry(id="a", text="x", embedding=[1.0])
        e2 = MemoryEntry(id="b", text="y", embedding=[1.0])
        e1.metadata["key"] = "val"
        assert "key" not in e2.metadata

    def test_custom_metadata(self):
        entry = MemoryEntry(id="c", text="z", embedding=[1.0], metadata={"src": "test"})
        assert entry.metadata["src"] == "test"


# ============================================================
#  _cosine_similarity
# ============================================================

class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert _cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector_a(self):
        assert _cosine_similarity([0, 0], [1, 1]) == 0.0

    def test_zero_vector_b(self):
        assert _cosine_similarity([1, 1], [0, 0]) == 0.0

    def test_both_zero(self):
        assert _cosine_similarity([0, 0], [0, 0]) == 0.0

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            _cosine_similarity([1, 2], [1, 2, 3])

    def test_single_element(self):
        assert _cosine_similarity([5], [5]) == pytest.approx(1.0)

    def test_negative_values(self):
        result = _cosine_similarity([-1, -2], [-1, -2])
        assert result == pytest.approx(1.0)


# ============================================================
#  InMemoryBackend
# ============================================================

class TestInMemoryBackend:
    def test_upsert_and_query(self):
        backend = InMemoryBackend(dim=3)
        entries = [
            MemoryEntry(id="cat", text="cat", embedding=[1, 0, 0]),
            MemoryEntry(id="dog", text="dog", embedding=[0, 1, 0]),
            MemoryEntry(id="fish", text="fish", embedding=[0, 0, 1]),
        ]
        for e in entries:
            backend.upsert(e)

        results = backend.query([1, 0, 0], k=2)
        assert len(results) == 2
        assert results[0][0] == "cat"
        assert results[0][1] == pytest.approx(1.0)

    def test_upsert_wrong_dim(self):
        backend = InMemoryBackend(dim=4)
        entry = MemoryEntry(id="x", text="x", embedding=[1, 0, 0])  # dim=3
        with pytest.raises(ValueError, match="dimension"):
            backend.upsert(entry)

    def test_query_wrong_dim(self):
        backend = InMemoryBackend(dim=4)
        backend.upsert(MemoryEntry(id="x", text="x", embedding=[1, 0, 0, 0]))
        with pytest.raises(ValueError, match="dimension"):
            backend.query([1, 0, 0], k=1)

    def test_upsert_replaces_existing(self):
        backend = InMemoryBackend(dim=2)
        backend.upsert(MemoryEntry(id="x", text="old", embedding=[1, 0]))
        backend.upsert(MemoryEntry(id="x", text="new", embedding=[0, 1]))
        assert backend.count() == 1

    def test_delete_existing(self):
        backend = InMemoryBackend(dim=2)
        backend.upsert(MemoryEntry(id="x", text="x", embedding=[1, 0]))
        assert backend.delete("x") is True
        assert backend.count() == 0

    def test_delete_nonexistent(self):
        backend = InMemoryBackend(dim=2)
        assert backend.delete("ghost") is False

    def test_count_empty(self):
        backend = InMemoryBackend(dim=2)
        assert backend.count() == 0

    def test_count_after_multiple_inserts(self):
        backend = InMemoryBackend(dim=2)
        for i in range(10):
            backend.upsert(MemoryEntry(id=f"id{i}", text="t", embedding=[float(i), 0]))
        assert backend.count() == 10

    def test_query_returns_metadata(self):
        backend = InMemoryBackend(dim=2)
        backend.upsert(MemoryEntry(
            id="x", text="x", embedding=[1, 0], metadata={"src": "test"}
        ))
        results = backend.query([1, 0], k=1)
        assert results[0][2] == {"src": "test"}

    def test_query_k_larger_than_entries(self):
        backend = InMemoryBackend(dim=2)
        backend.upsert(MemoryEntry(id="x", text="x", embedding=[1, 0]))
        results = backend.query([1, 0], k=10)
        assert len(results) == 1

    def test_query_returns_copy_of_metadata(self):
        """Query should return a dict copy, not the internal reference."""
        backend = InMemoryBackend(dim=2)
        backend.upsert(MemoryEntry(
            id="x", text="x", embedding=[1, 0], metadata={"key": "val"}
        ))
        results = backend.query([1, 0], k=1)
        results[0][2]["new_key"] = "new"
        # Original should be unchanged
        assert "new_key" not in backend._entries["x"].metadata

    def test_query_orders_by_score(self):
        backend = InMemoryBackend(dim=2)
        backend.upsert(MemoryEntry(id="a", text="a", embedding=[1, 0]))
        backend.upsert(MemoryEntry(id="b", text="b", embedding=[0.9, 0.1]))
        backend.upsert(MemoryEntry(id="c", text="c", embedding=[0, 1]))
        results = backend.query([1, 0], k=3)
        assert results[0][0] == "a"
        assert results[1][0] == "b"
        assert results[2][0] == "c"

    def test_custom_dim(self):
        backend = InMemoryBackend(dim=128)
        entry = MemoryEntry(id="x", text="x", embedding=[0.1] * 128)
        backend.upsert(entry)
        assert backend.count() == 1


# ============================================================
#  SQLiteVecBackend (conditional on sqlite_vec availability)
# ============================================================

@pytest.fixture
def sqlite_backend(tmp_path):
    """Create a SQLiteVecBackend, skip if sqlite_vec not installed."""
    try:
        backend = SQLiteVecBackend(path=tmp_path / "test.db", dim=4)
        return backend
    except ImportError:
        pytest.skip("sqlite_vec not installed")


class TestSQLiteVecBackend:
    def test_construction(self, sqlite_backend):
        assert sqlite_backend.count() == 0

    def test_upsert_and_count(self, sqlite_backend):
        entry = MemoryEntry(id="x", text="hello", embedding=[1.0, 0, 0, 0], metadata={"k": "v"})
        sqlite_backend.upsert(entry)
        assert sqlite_backend.count() == 1

    def test_upsert_wrong_dim(self, sqlite_backend):
        entry = MemoryEntry(id="x", text="x", embedding=[1, 0, 0])  # dim=3
        with pytest.raises(ValueError):
            sqlite_backend.upsert(entry)

    def test_query_wrong_dim(self, sqlite_backend):
        with pytest.raises(ValueError):
            sqlite_backend.query([1, 0, 0], k=1)

    def test_delete_existing(self, sqlite_backend):
        sqlite_backend.upsert(MemoryEntry(id="x", text="x", embedding=[1, 0, 0, 0]))
        assert sqlite_backend.delete("x") is True
        assert sqlite_backend.count() == 0

    def test_delete_nonexistent(self, sqlite_backend):
        assert sqlite_backend.delete("ghost") is False

    def test_upsert_replaces(self, sqlite_backend):
        """Upserting same doc_id should replace, not duplicate."""
        sqlite_backend.upsert(MemoryEntry(id="x", text="old", embedding=[1, 0, 0, 0]))
        sqlite_backend.upsert(MemoryEntry(id="x", text="new", embedding=[0, 1, 0, 0]))
        assert sqlite_backend.count() == 1

    def test_query_returns_results(self, sqlite_backend):
        entries = [
            MemoryEntry(id="a", text="cat", embedding=[1, 0, 0, 0]),
            MemoryEntry(id="b", text="dog", embedding=[0, 1, 0, 0]),
        ]
        for e in entries:
            sqlite_backend.upsert(e)
        results = sqlite_backend.query([1, 0, 0, 0], k=1)
        assert len(results) >= 1
        assert results[0][0] == "a"

    def test_persistence_across_connection(self, tmp_path):
        """Data should persist when using a file path."""
        try:
            db_path = tmp_path / "persist.db"
            backend1 = SQLiteVecBackend(path=db_path, dim=4)
            backend1.upsert(MemoryEntry(id="x", text="persistent", embedding=[1, 0, 0, 0]))
            backend1._db.close()

            backend2 = SQLiteVecBackend(path=db_path, dim=4)
            assert backend2.count() == 1
        except ImportError:
            pytest.skip("sqlite_vec not installed")


# ============================================================
#  MemoryIndex
# ============================================================

class TestMemoryIndex:
    def test_default_uses_in_memory_when_no_sqlite(self):
        """When sqlite_vec is unavailable, should fall back to InMemoryBackend."""
        with patch("exocortex.memory.SQLiteVecBackend", side_effect=ImportError):
            idx = MemoryIndex()
        assert isinstance(idx.backend, InMemoryBackend)

    def test_explicit_in_memory_backend(self):
        backend = InMemoryBackend(dim=3)
        idx = MemoryIndex(backend=backend)
        assert idx.backend is backend

    def test_upsert_with_explicit_embedding(self):
        idx = MemoryIndex(backend=InMemoryBackend(dim=3))
        entry = idx.upsert("doc1", "hello", embedding=[1, 0, 0])
        assert entry.id == "doc1"
        assert entry.embedding == [1, 0, 0]

    def test_upsert_computes_embedding(self):
        """When no embedding provided, should call embed_fn."""
        called = []

        def mock_embed(text):
            called.append(text)
            return [1.0, 0.0, 0.0]

        idx = MemoryIndex(backend=InMemoryBackend(dim=3), embed_fn=mock_embed)
        idx.upsert("doc1", "hello world")
        assert "hello world" in called

    def test_upsert_with_metadata(self):
        idx = MemoryIndex(backend=InMemoryBackend(dim=3))
        idx.upsert("doc1", "hello", metadata={"src": "test"}, embedding=[1, 0, 0])
        results = idx.query(embedding=[1, 0, 0], k=1)
        assert results[0][2] == {"src": "test"}

    def test_query_by_text(self):
        idx = MemoryIndex(backend=InMemoryBackend(dim=3), embed_fn=lambda t: [1, 0, 0])
        idx.upsert("a", "cat", embedding=[1, 0, 0])
        results = idx.query(text="cat", k=1)
        assert len(results) >= 1

    def test_query_by_embedding(self):
        idx = MemoryIndex(backend=InMemoryBackend(dim=3))
        idx.upsert("a", "cat", embedding=[1, 0, 0])
        results = idx.query(embedding=[1, 0, 0], k=1)
        assert results[0][0] == "a"

    def test_query_requires_text_or_embedding(self):
        idx = MemoryIndex(backend=InMemoryBackend(dim=3))
        with pytest.raises(ValueError, match="Provide either"):
            idx.query()

    def test_delete(self):
        idx = MemoryIndex(backend=InMemoryBackend(dim=3))
        idx.upsert("a", "cat", embedding=[1, 0, 0])
        assert idx.delete("a") is True
        assert idx.count() == 0

    def test_count(self):
        idx = MemoryIndex(backend=InMemoryBackend(dim=3))
        idx.upsert("a", "x", embedding=[1, 0, 0])
        idx.upsert("b", "y", embedding=[0, 1, 0])
        assert idx.count() == 2

    def test_custom_dim(self):
        backend = InMemoryBackend(dim=10)
        idx = MemoryIndex(backend=backend, dim=10)
        assert idx.dim == 10


# ============================================================
#  write_outcome
# ============================================================

class TestWriteOutcome:
    def test_write_outcome_in_memory(self):
        idx = MemoryIndex(backend=InMemoryBackend(dim=3))
        idx.upsert("doc1", "hello", embedding=[1, 0, 0])
        result = idx.write_outcome("doc1", {"action": "tested", "success": True})
        assert result is not None
        assert "outcomes" in result.metadata
        assert result.metadata["outcomes"][0]["action"] == "tested"

    def test_write_outcome_appends_multiple(self):
        idx = MemoryIndex(backend=InMemoryBackend(dim=3))
        idx.upsert("doc1", "hello", embedding=[1, 0, 0])
        idx.write_outcome("doc1", {"r": 1})
        idx.write_outcome("doc1", {"r": 2})
        entry = idx.backend._entries["doc1"]
        assert len(entry.metadata["outcomes"]) == 2
        assert entry.metadata["outcomes"][1]["r"] == 2

    def test_write_outcome_nonexistent_doc(self):
        idx = MemoryIndex(backend=InMemoryBackend(dim=3))
        result = idx.write_outcome("ghost", {"x": 1})
        assert result is None

    def test_write_outcome_preserves_existing_metadata(self):
        idx = MemoryIndex(backend=InMemoryBackend(dim=3))
        idx.upsert("doc1", "hello", embedding=[1, 0, 0], metadata={"orig": True})
        idx.write_outcome("doc1", {"r": 1})
        entry = idx.backend._entries["doc1"]
        assert entry.metadata["orig"] is True
        assert "outcomes" in entry.metadata

    def test_write_outcome_sqlite(self, tmp_path):
        """Test write_outcome with SQLiteVecBackend if available."""
        try:
            backend = SQLiteVecBackend(path=tmp_path / "wo.db", dim=3)
        except ImportError:
            pytest.skip("sqlite_vec not installed")

        idx = MemoryIndex(backend=backend)
        idx.upsert("doc1", "hello", embedding=[1, 0, 0])
        result = idx.write_outcome("doc1", {"action": "done"})
        assert result is not None
        assert result.metadata["outcomes"][0]["action"] == "done"

    def test_write_outcome_sqlite_nonexistent(self, tmp_path):
        try:
            backend = SQLiteVecBackend(path=tmp_path / "wo2.db", dim=3)
        except ImportError:
            pytest.skip("sqlite_vec not installed")

        idx = MemoryIndex(backend=backend)
        result = idx.write_outcome("ghost", {"x": 1})
        assert result is None

    def test_write_outcome_unknown_backend_type(self):
        """write_outcome should return None for unknown backend types."""
        mock_backend = MagicMock(spec=VectorBackend)
        idx = MemoryIndex(backend=mock_backend)
        result = idx.write_outcome("doc1", {"x": 1})
        assert result is None


# ============================================================
#  VectorBackend ABC
# ============================================================

class TestVectorBackendABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            VectorBackend()

    def test_subclass_must_implement_all_methods(self):
        class Partial(VectorBackend):
            def upsert(self, entry):
                pass
        with pytest.raises(TypeError):
            Partial()


# ============================================================
#  Integration: MemoryIndex end-to-end
# ============================================================

class TestMemoryIndexIntegration:
    def test_full_lifecycle(self):
        """Add, query, update, delete cycle."""
        idx = MemoryIndex(backend=InMemoryBackend(dim=3))

        # Add
        idx.upsert("a", "cats", embedding=[1, 0, 0], metadata={"tag": "animal"})
        idx.upsert("b", "dogs", embedding=[0, 1, 0], metadata={"tag": "animal"})
        idx.upsert("c", "rocks", embedding=[0, 0, 1], metadata={"tag": "mineral"})

        # Query
        results = idx.query(embedding=[1, 0.1, 0], k=2)
        assert len(results) == 2
        assert results[0][0] == "a"  # closest to [1,0,0]

        # Update metadata via write_outcome
        idx.write_outcome("a", {"clicked": True})

        # Delete
        assert idx.delete("c") is True
        assert idx.count() == 2

    def test_repeated_upsert_same_id(self):
        """Repeated upserts with same ID should not grow count."""
        idx = MemoryIndex(backend=InMemoryBackend(dim=3))
        for i in range(5):
            idx.upsert("same", f"text{i}", embedding=[1, 0, 0])
        assert idx.count() == 1

    def test_large_batch(self):
        idx = MemoryIndex(backend=InMemoryBackend(dim=2))
        for i in range(100):
            idx.upsert(f"id{i}", f"doc{i}", embedding=[float(i % 10), float(i // 10)])
        assert idx.count() == 100
        results = idx.query(embedding=[5, 5], k=5)
        assert len(results) == 5
