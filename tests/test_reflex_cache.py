import json
import tempfile
from pathlib import Path

import pytest

from exocortex import NailReflex, ReflexCache


def test_exact_lookup_returns_response():
    cache = ReflexCache()
    cache.store("what is 2+2", "4", confidence=0.9)
    reflex = cache.lookup("what is 2+2")
    assert reflex is not None
    assert reflex.response == "4"


def test_lookup_below_threshold_is_none():
    cache = ReflexCache(confidence_threshold=0.85)
    cache.store("what is 2+2", "4", confidence=0.5)
    assert cache.lookup("what is 2+2") is None


def test_confidence_update_success_and_failure():
    cache = ReflexCache()
    rid = cache.store("q", "a", confidence=0.5)

    # One success should raise confidence but stay below reflex threshold.
    c1 = cache.update_confidence(rid, success=True)
    assert c1 == pytest.approx(0.5 + 0.05 * (1 - 0.5))

    # Move to a high confidence and then fail twice.
    cache._by_id[rid].confidence = 0.9
    c2 = cache.update_confidence(rid, success=False)
    assert c2 == pytest.approx(0.9 - 0.10 * 0.9)
    c3 = cache.update_confidence(rid, success=False)
    assert c3 == pytest.approx(c2 - 0.10 * c2)

    # Clamp
    cache._by_id[rid].confidence = 0.01
    assert cache.update_confidence(rid, success=False) == 0.05
    cache._by_id[rid].confidence = 0.99
    assert cache.update_confidence(rid, success=True) == 0.95


def test_max_consecutive_uses_escape_hatch():
    cache = ReflexCache()
    rid = cache.store("q", "a", confidence=0.9, max_consecutive_uses=3)

    # Three hits succeed.
    assert cache.lookup("q") is not None
    assert cache.lookup("q") is not None
    assert cache.lookup("q") is not None
    # Fourth hit triggers the escape hatch.
    assert cache.lookup("q") is None
    # Counter is reset, so the next hit succeeds again.
    assert cache.lookup("q") is not None


def test_vector_nearest_neighbour_fallback():
    # Use a deterministic embedder so two different strings share an embedding.
    shared = [0.1] * 384

    def embed(_text: str):
        return list(shared)

    cache = ReflexCache(embed_fn=embed)
    rid = cache.store("original signature", "answer", confidence=0.9)

    # Exact hash misses, but vector NN should hit.
    reflex = cache.lookup("different signature")
    assert reflex is not None
    assert reflex.id == rid


def test_save_and_load_directory(tmp_path: Path):
    cache = ReflexCache()
    rid = cache.store("q", "a", confidence=0.88, metadata={"domain": "test"})
    cache.save(tmp_path)

    loaded = ReflexCache()
    count = loaded.load(tmp_path)
    assert count == 1
    assert loaded.size() == 1
    reflex = loaded.lookup("q")
    assert reflex is not None
    assert reflex.id == rid
    assert reflex.metadata["domain"] == "test"


def test_save_and_load_single_file(tmp_path: Path):
    cache = ReflexCache()
    cache.store("q", "a", confidence=0.88)
    file_path = tmp_path / "reflexes.json"
    cache.save(file_path)

    assert file_path.exists()
    data = json.loads(file_path.read_text())
    assert len(data) == 1

    loaded = ReflexCache()
    loaded.load(file_path)
    assert loaded.lookup("q") is not None


def test_stats_report():
    cache = ReflexCache()
    assert cache.stats()["total"] == 0
    cache.store("q1", "a1", confidence=0.9)
    cache.store("q2", "a2", confidence=0.5)
    stats = cache.stats()
    assert stats["total"] == 2
    assert stats["active"] == 1
