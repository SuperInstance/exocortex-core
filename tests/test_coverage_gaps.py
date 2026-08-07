"""Tests targeting specific uncovered code paths to close coverage gaps.

Focuses on:
- ReflexCache: vector-NN blocked by escape hatch, reset_consecutive on
  existing id, save to directory with empty cache, load from .nail files.
- VoiceGate: action starting with "cloud" (line 57).
- MemoryIndex: custom sqlite_path argument, _default_backend with explicit path.
- hash_embedding: exported from __init__.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from exocortex import (
    ReflexCache,
    NailReflex,
    VoiceGate,
    Trigger,
    RouteTarget,
    MemoryIndex,
    hash_embedding,
)
from exocortex.reflex_cache import _cosine_similarity


# ============================================================
# ReflexCache — uncovered lines
# ============================================================

class TestReflexCacheVectorBlocked:
    """Vector NN fallback when exact match is blocked by escape hatch."""

    def test_vector_nn_blocked_by_escape_hatch(self):
        """When exact-match reflex hits escape hatch, vector NN to the
        same reflex should also be blocked (returns None, not the same reflex).
        """
        shared = [0.1] * 384

        def embed(_text: str):
            return list(shared)

        cache = ReflexCache(embed_fn=embed)
        rid = cache.store("original", "answer", confidence=0.9, max_consecutive_uses=2)

        # Two hits exhaust the escape hatch via exact match.
        assert cache.lookup("original") is not None
        assert cache.lookup("original") is not None
        # Third consecutive: exact hash matches but escape hatch blocks.
        # The code explicitly returns None instead of falling through to NN.
        assert cache.lookup("original") is None

    def test_vector_nn_below_threshold_returns_none(self):
        """When the nearest vector is below 0.85 cosine similarity, return None."""

        def embed(text: str):
            # Produce different embeddings for different texts.
            return hash_embedding(text, dim=384)

        cache = ReflexCache(embed_fn=embed)
        cache.store("alpha topic", "response a", confidence=0.9)

        # A completely different string should have low cosine similarity.
        result = cache.lookup("zzzzzzz completely different")
        assert result is None

    def test_vector_nn_skips_below_confidence(self):
        """Reflexes below the confidence threshold are excluded from NN search."""

        def embed(_text: str):
            return [0.5] * 384

        cache = ReflexCache(embed_fn=embed, confidence_threshold=0.85)
        cache.store("known thing", "response", confidence=0.5)  # below threshold

        # Vector NN should not return a below-threshold reflex.
        assert cache.lookup("similar thing") is None


class TestResetConsecutive:
    def test_reset_consecutive_existing_id(self):
        """reset_consecutive should reset the counter for an existing reflex."""
        cache = ReflexCache()
        rid = cache.store("q", "a", confidence=0.9, max_consecutive_uses=3)

        # Exhaust the escape hatch.
        assert cache.lookup("q") is not None
        assert cache.lookup("q") is not None
        assert cache.lookup("q") is not None
        # Now blocked.
        assert cache.lookup("q") is None

        # Reset should allow usage again.
        cache.reset_consecutive(rid)
        assert cache.lookup("q") is not None


class TestSaveEmptyCache:
    def test_save_empty_cache_to_json(self, tmp_path: Path):
        """Saving an empty cache should produce a valid JSON file."""
        cache = ReflexCache()
        out = tmp_path / "empty.json"
        cache.save(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data == []

    def test_save_empty_cache_to_directory(self, tmp_path: Path):
        """Saving an empty cache to a directory should not fail."""
        cache = ReflexCache()
        out_dir = tmp_path / "nails"
        cache.save(out_dir)
        assert out_dir.is_dir()
        # No .nail files.
        assert list(out_dir.glob("*.nail")) == []

    def test_load_empty_json(self, tmp_path: Path):
        """Loading an empty JSON file should work."""
        out = tmp_path / "empty.json"
        out.write_text("[]")
        cache = ReflexCache()
        count = cache.load(out)
        assert count == 0
        assert cache.size() == 0


class TestLoadFromNailFiles:
    def test_load_multiple_nail_files(self, tmp_path: Path):
        """Loading from a directory with multiple .nail files."""
        cache = ReflexCache()
        cache.store("question one", "answer one", confidence=0.9)
        cache.store("question two", "answer two", confidence=0.88)
        cache.save(tmp_path)

        loaded = ReflexCache()
        count = loaded.load(tmp_path)
        assert count == 2
        assert loaded.lookup("question one") is not None
        assert loaded.lookup("question two") is not None


class TestReflexCacheConfidenceRound:
    def test_update_confidence_rounds_to_six_decimals(self):
        """update_confidence should round to 6 decimal places."""
        cache = ReflexCache()
        rid = cache.store("q", "a", confidence=0.7)
        new_c = cache.update_confidence(rid, success=True)
        # 0.7 + 0.05 * (1 - 0.7) = 0.7 + 0.015 = 0.715
        assert new_c == pytest.approx(0.715)
        # Verify the stored value is rounded.
        assert cache._by_id[rid].confidence == round(0.715, 6)


class TestSaveJsonImplicitPath:
    def test_save_json_nonexistent_path(self, tmp_path: Path):
        """save() with a .json path that doesn't exist yet should create it."""
        cache = ReflexCache()
        cache.store("q", "a", confidence=0.9)
        nested = tmp_path / "subdir" / "reflexes.json"
        cache.save(nested)
        assert nested.exists()
        data = json.loads(nested.read_text())
        assert len(data) == 1


# ============================================================
# VoiceGate — uncovered line 57: action starting with "cloud"
# ============================================================

class TestVoiceGateCloudAction:
    def test_cloud_action_routes_to_cloud(self):
        """A trigger with action 'cloud:...' should route to CLOUD."""
        gate = VoiceGate()
        gate.register(
            Trigger(
                category="complex",
                phrase="explain quantum physics",
                action="cloud:deep_reasoning",
                match_type="exact",
            )
        )
        decision = gate.classify("explain quantum physics")
        assert decision.target == RouteTarget.CLOUD
        assert decision.confidence == 1.0

    def test_cloud_action_substring_match(self):
        gate = VoiceGate()
        gate.register(
            Trigger(
                category="complex",
                phrase="quantum",
                action="cloud",
                match_type="substring",
            )
        )
        decision = gate.classify("tell me about quantum mechanics")
        assert decision.target == RouteTarget.CLOUD
        assert decision.confidence == 0.85


# ============================================================
# hash_embedding export from __init__
# ============================================================

class TestHashEmbeddingExport:
    def test_hash_embedding_importable_from_package(self):
        """hash_embedding should be importable from the top-level package."""
        v = hash_embedding("test")
        assert isinstance(v, list)
        assert len(v) == 384

    def test_hash_embedding_custom_dim(self):
        v = hash_embedding("test", dim=128)
        assert len(v) == 128

    def test_hash_embedding_invalid_dim_negative(self):
        with pytest.raises(ValueError, match="dim"):
            hash_embedding("test", dim=-4)

    def test_hash_embedding_invalid_dim_non_multiple(self):
        with pytest.raises(ValueError, match="dim"):
            hash_embedding("test", dim=7)


# ============================================================
# MemoryIndex — custom sqlite_path path
# ============================================================

class TestMemoryIndexSqlitePath:
    def test_custom_sqlite_path_falls_back_gracefully(self, tmp_path: Path):
        """When sqlite_vec is not installed, providing sqlite_path should
        still result in InMemoryBackend fallback."""
        from unittest.mock import patch
        with patch("exocortex.memory.SQLiteVecBackend", side_effect=ImportError):
            idx = MemoryIndex(sqlite_path=tmp_path / "test.db")
        from exocortex.memory import InMemoryBackend
        assert isinstance(idx.backend, InMemoryBackend)


# ============================================================
# Cosine similarity in reflex_cache module
# ============================================================

class TestReflexCacheCosineUtil:
    def test_cosine_similarity_reflex_cache_module(self):
        """The _cosine_similarity in reflex_cache should work correctly."""
        assert _cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
        assert _cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_lookup_on_empty_cache_returns_none(self):
        """Vector NN on an empty cache should return None (early-return path)."""
        cache = ReflexCache()
        assert cache.lookup("anything") is None


# ============================================================
# Integration: full pipeline
# ============================================================

class TestPipelineIntegration:
    def test_reflex_cache_after_distillation_confidence_update(self):
        """Simulate distillation compiling a reflex, then updating confidence."""
        cache = ReflexCache()
        rid = cache.store("compiled lesson", "use --!strict", confidence=0.6)

        # Verify the reflex is below threshold initially.
        assert cache.lookup("compiled lesson") is None

        # Simulate successful outcomes raising confidence above threshold.
        # At +5%*(1-c) per success, it takes ~20 iterations from 0.6 to reach 0.85.
        for _ in range(25):
            cache.update_confidence(rid, success=True)

        # Now it should be retrievable.
        reflex = cache.lookup("compiled lesson")
        assert reflex is not None
        assert reflex.confidence >= cache.confidence_threshold

    def test_voice_gate_to_router_pipeline(self):
        """Voice gate output feeds into router: unknown transcript → CLOUD."""
        gate = VoiceGate()
        decision = gate.classify("some unknown complex query")
        assert decision.target == RouteTarget.CLOUD

        # The router should also route to CLOUD for unknown embeddings.
        from exocortex import ExoRouter
        router = ExoRouter()
        route = router.route([0.1] * 384)
        assert route.target == RouteTarget.CLOUD
