import pytest

from exocortex.memory import InMemoryBackend, MemoryIndex, SQLiteVecBackend


def test_upsert_and_query_in_memory():
    index = MemoryIndex(backend=InMemoryBackend(dim=384))
    index.upsert("doc1", "the quick brown fox")
    index.upsert("doc2", "lazy dog sleeping")
    results = index.query("quick fox", k=2)
    assert len(results) == 2
    ids = [r[0] for r in results]
    assert "doc1" in ids


def test_delete_removes_entry():
    index = MemoryIndex(backend=InMemoryBackend(dim=384))
    index.upsert("doc1", "the quick brown fox")
    assert index.delete("doc1") is True
    assert index.count() == 0
    assert index.delete("missing") is False


def test_write_outcome_appends_metadata():
    index = MemoryIndex(backend=InMemoryBackend(dim=384))
    index.upsert("doc1", "task started")
    entry = index.write_outcome("doc1", {"success": True, "quality": 0.9})
    assert entry is not None
    assert len(entry.metadata["outcomes"]) == 1
    assert entry.metadata["outcomes"][0]["success"] is True


def test_sqlite_vec_unavailable_falls_back():
    try:
        backend = SQLiteVecBackend(dim=384)
    except ImportError:
        backend = None

    if backend is None:
        # The public index should degrade to in-memory when sqlite_vec is missing.
        index = MemoryIndex()
        assert isinstance(index.backend, InMemoryBackend)
        index.upsert("doc1", "fallback works")
        assert index.count() == 1
    else:
        pytest.skip("sqlite_vec is installed; fallback path not exercised")


def test_query_requires_text_or_embedding():
    index = MemoryIndex(backend=InMemoryBackend(dim=384))
    with pytest.raises(ValueError):
        index.query()
