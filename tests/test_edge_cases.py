"""Additional exocortex tests — edge cases for reflex cache, voice gate, and memory."""

import json
import pytest
from pathlib import Path

from exocortex import ReflexCache, VoiceGate, MemoryIndex
from exocortex.reflex_cache import NailReflex, _normalize, _sig_hash, _cosine_similarity
from exocortex.voice_gate import Trigger, VoiceDecision, RouteTarget, URGENCY_WORDS
from exocortex.memory import InMemoryBackend, MemoryEntry
from exocortex._embed import hash_embedding
from exocortex.bond import (
    award_bond, tier_for, tier_name, BondGate, BOND_POINTS,
    TIER_THRESHOLDS, TIER_NAMES, _event_value, _tier_floor,
)
from exocortex.distiller import (
    Evaluation, _score_response, _tokens, _unique_bigrams,
    stage_evaluate, stage_distill,
)


# ---------------------------------------------------------------------------
# ReflexCache edge cases
# ---------------------------------------------------------------------------

class TestReflexCacheEdgeCases:
    def test_normalize_collapses_whitespace(self):
        assert _normalize("  Hello   World  ") == "hello world"
        assert _normalize("HELLO") == "hello"
        assert _normalize("\t\ntab\n") == "tab"

    def test_normalize_empty_string(self):
        assert _normalize("") == ""

    def test_sig_hash_deterministic(self):
        assert _sig_hash("test") == _sig_hash("test")
        assert _sig_hash("Test") == _sig_hash("test")  # normalized
        assert _sig_hash("test") != _sig_hash("other")
        assert len(_sig_hash("x")) == 16

    def test_store_returns_valid_id(self):
        cache = ReflexCache()
        rid = cache.store("q", "a", confidence=0.9)
        assert len(rid) == 16  # secrets.token_hex(8)

    def test_store_clamps_confidence(self):
        cache = ReflexCache()
        rid_high = cache.store("q1", "a", confidence=99.0)
        rid_low = cache.store("q2", "a", confidence=-5.0)
        assert cache._by_id[rid_high].confidence == 0.95
        assert cache._by_id[rid_low].confidence == 0.05

    def test_store_with_custom_id(self):
        cache = ReflexCache()
        rid = cache.store("q", "a", confidence=0.9, reflex_id="my-custom-id")
        assert rid == "my-custom-id"

    def test_update_confidence_nonexistent(self):
        cache = ReflexCache()
        assert cache.update_confidence("ghost", success=True) is None

    def test_reset_consecutive_nonexistent(self):
        cache = ReflexCache()
        # Should not raise
        cache.reset_consecutive("ghost")

    def test_load_nonexistent_path(self):
        cache = ReflexCache()
        with pytest.raises(FileNotFoundError):
            cache.load("/nonexistent/path/to/nail/files")

    def test_reflex_post_init_rejects_empty_id(self):
        with pytest.raises(ValueError):
            NailReflex(id="", situation="s", match_key="k", response="r",
                       confidence=0.5, source="test", max_consecutive_uses=10)

    def test_cosine_similarity_basic(self):
        assert _cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
        assert _cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
        assert _cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_cosine_similarity_zero_vector(self):
        assert _cosine_similarity([0, 0], [1, 0]) == 0.0

    def test_cosine_similarity_mismatched_dims(self):
        with pytest.raises(ValueError):
            _cosine_similarity([1, 2], [1, 2, 3])

    def test_confidence_threshold_boundary(self):
        """A reflex exactly at threshold should be usable."""
        cache = ReflexCache(confidence_threshold=0.85)
        rid = cache.store("q", "a", confidence=0.85)
        assert cache.lookup("q") is not None

    def test_confidence_threshold_just_below(self):
        cache = ReflexCache(confidence_threshold=0.85)
        rid = cache.store("q", "a", confidence=0.84)
        assert cache.lookup("q") is None

    def test_stats_empty(self):
        cache = ReflexCache()
        stats = cache.stats()
        assert stats["total"] == 0
        assert stats["avg_confidence"] == 0.0

    def test_size_after_store(self):
        cache = ReflexCache()
        cache.store("q1", "a1", confidence=0.9)
        cache.store("q2", "a2", confidence=0.9)
        assert cache.size() == 2

    def test_repeated_lookup_increments_consecutive(self):
        cache = ReflexCache()
        rid = cache.store("q", "a", confidence=0.9, max_consecutive_uses=5)
        for _ in range(5):
            assert cache.lookup("q") is not None
        # 6th should be blocked
        assert cache.lookup("q") is None


# ---------------------------------------------------------------------------
# VoiceGate edge cases
# ---------------------------------------------------------------------------

class TestVoiceGateEdgeCases:
    def test_list_triggers_returns_copy(self):
        gate = VoiceGate()
        gate.register(Trigger(category="test", phrase="hello", action="local"))
        triggers = gate.list_triggers()
        triggers.clear()
        # Original should be unaffected
        assert len(gate.list_triggers()) == 1

    def test_empty_transcript(self):
        gate = VoiceGate()
        decision = gate.classify("")
        assert decision.target == RouteTarget.CLOUD
        assert decision.confidence == 0.2

    def test_whitespace_transcript(self):
        gate = VoiceGate()
        decision = gate.classify("   \n\t  ")
        assert decision.target == RouteTarget.CLOUD

    def test_case_insensitive_matching(self):
        gate = VoiceGate()
        gate.register(Trigger(category="test", phrase="Hello World", action="local", match_type="exact"))
        decision = gate.classify("HELLO WORLD")
        assert decision.target == RouteTarget.LOCAL

    def test_multiple_triggers_first_match_wins(self):
        gate = VoiceGate()
        gate.register(Trigger(category="a", phrase="hello", action="reflex:a"))
        gate.register(Trigger(category="b", phrase="hello", action="local:b"))
        decision = gate.classify("hello")
        assert decision.trigger.category == "a"

    def test_urgency_word_set_not_empty(self):
        assert len(URGENCY_WORDS) > 0
        assert "stop" in URGENCY_WORDS

    def test_target_for_action_defaults_to_local(self):
        from exocortex.voice_gate import _target_for_action
        assert _target_for_action("unknown_action") == RouteTarget.LOCAL

    def test_substring_match_in_longer_sentence(self):
        gate = VoiceGate()
        gate.register(Trigger(category="weather", phrase="weather", action="local", match_type="substring"))
        decision = gate.classify("hey what's the weather like in anchorage")
        assert decision.target == RouteTarget.LOCAL
        assert decision.matched_phrase == "weather"

    def test_exact_match_does_not_match_substring(self):
        gate = VoiceGate()
        gate.register(Trigger(category="test", phrase="go", action="reflex:go", match_type="exact"))
        # "going" should not trigger an exact match on "go"
        decision = gate.classify("going home")
        assert decision.target == RouteTarget.CLOUD

    def test_decision_is_frozen(self):
        """VoiceDecision is a frozen dataclass."""
        decision = VoiceDecision(target=RouteTarget.CLOUD, confidence=0.5, urgent=False)
        with pytest.raises(AttributeError):
            decision.target = RouteTarget.LOCAL


# ---------------------------------------------------------------------------
# MemoryIndex edge cases
# ---------------------------------------------------------------------------

class TestMemoryEdgeCases:
    def test_upsert_overwrites_existing(self):
        index = MemoryIndex(backend=InMemoryBackend(dim=384))
        index.upsert("doc1", "first version")
        index.upsert("doc1", "second version")
        assert index.count() == 1
        results = index.query("second", k=1)
        assert results[0][0] == "doc1"

    def test_query_returns_metadata(self):
        index = MemoryIndex(backend=InMemoryBackend(dim=384))
        index.upsert("doc1", "text", metadata={"source": "test"})
        results = index.query("text", k=1)
        assert results[0][2]["source"] == "test"

    def test_write_outcome_nonexistent_returns_none(self):
        index = MemoryIndex(backend=InMemoryBackend(dim=384))
        result = index.write_outcome("ghost", {"x": 1})
        assert result is None

    def test_write_outcome_multiple_appends(self):
        index = MemoryIndex(backend=InMemoryBackend(dim=384))
        index.upsert("doc1", "text")
        index.write_outcome("doc1", {"success": True})
        index.write_outcome("doc1", {"success": False})
        entry = index.write_outcome("doc1", {"success": True})
        assert len(entry.metadata["outcomes"]) == 3

    def test_in_memory_backend_dimension_mismatch(self):
        backend = InMemoryBackend(dim=128)
        with pytest.raises(ValueError, match="dimension"):
            backend.upsert(MemoryEntry(id="x", text="t", embedding=[0.1] * 64))

    def test_in_memory_query_dimension_mismatch(self):
        backend = InMemoryBackend(dim=128)
        with pytest.raises(ValueError, match="dimension"):
            backend.query([0.1] * 64)

    def test_hash_embedding_validates_dim(self):
        with pytest.raises(ValueError):
            hash_embedding("test", dim=0)
        with pytest.raises(ValueError):
            hash_embedding("test", dim=5)  # not multiple of 4

    def test_hash_embedding_deterministic(self):
        v1 = hash_embedding("hello world")
        v2 = hash_embedding("hello world")
        assert v1 == v2

    def test_hash_embedding_different_texts_differ(self):
        v1 = hash_embedding("hello")
        v2 = hash_embedding("world")
        assert v1 != v2

    def test_hash_embedding_value_range(self):
        v = hash_embedding("test text")
        assert all(-1.0 <= x <= 1.0 for x in v)


# ---------------------------------------------------------------------------
# Bond edge cases
# ---------------------------------------------------------------------------

class TestBondEdgeCases:
    def test_tier_name_boundaries(self):
        assert tier_name(0) == "Hired"
        assert tier_name(10) == "Working Together"
        assert tier_name(30) == "Trusted"
        assert tier_name(70) == "Crew"
        assert tier_name(150) == "The Yard"

    def test_negative_event_at_floor_zero(self):
        """At tier 0 (floor=0), negative events can't go below 0."""
        new = award_bond(0, "blind_delete", {})
        assert new == 0  # floor of tier 0 is 0

    def test_cap_exempt_finished_hook(self):
        """finished_hook bypasses the session cap for positive points."""
        # With 0 prior events, full base value of 5 is awarded.
        events = {}
        new = award_bond(50, "finished_hook", events, session_cap=10)
        assert new == 55  # 50 + 5, uncapped by session_cap of 10

    def test_event_value_diminishing_returns(self):
        # First two uses are full value
        assert _event_value("manual_build", 0) == 3
        assert _event_value("manual_build", 1) == 3
        # Third+ is halved
        assert _event_value("manual_build", 2) == 1

    def test_event_value_negative_halves_toward_zero(self):
        # blind_delete = -1; halved via -((-base)//2) = -(1//2) = 0
        assert _event_value("blind_delete", 0) == -1  # first use, full value
        assert _event_value("blind_delete", 2) == 0   # halved rounds to 0

    def test_bond_gate_register_action(self):
        gate = BondGate(bond_level=0)
        gate.register_action("custom_action", 2)
        assert gate.required_tier("custom_action") == 2
        assert gate.allowed("custom_action") is False
        gate.update_bond_level(35)
        assert gate.allowed("custom_action") is True

    def test_bond_gate_update_changes_tier(self):
        gate = BondGate(bond_level=0)
        assert gate.tier == 0
        gate.update_bond_level(35)
        assert gate.tier == 2

    def test_all_tier_names_match_thresholds(self):
        assert len(TIER_NAMES) == len(TIER_THRESHOLDS)

    def test_all_bond_points_have_entries(self):
        for event, pts in BOND_POINTS.items():
            assert isinstance(event, str)
            assert isinstance(pts, int)


# ---------------------------------------------------------------------------
# Distiller edge cases
# ---------------------------------------------------------------------------

class TestDistillerEdgeCases:
    def test_tokens_extracts_alphanumeric(self):
        result = _tokens("Hello, World! 123")
        assert result == ["hello", "world", "123"]

    def test_tokens_empty_string(self):
        assert _tokens("") == []

    def test_unique_bigrams(self):
        result = _unique_bigrams(["a", "b", "c"])
        assert ("a", "b") in result
        assert ("b", "c") in result
        assert ("a", "c") not in result

    def test_score_response_empty_text(self):
        scores = _score_response("", "reference")
        assert scores.novelty == 0.0

    def test_score_response_identical_texts(self):
        text = "the quick brown fox"
        scores = _score_response(text, text)
        # No novel bigrams when compared to itself
        assert scores.novelty == 0.0

    def test_evaluation_average(self):
        e = Evaluation(novelty=0.5, specificity=0.3, engagement=0.8, spatial=0.2)
        assert e.average == pytest.approx(0.45)

    def test_stage_evaluate_zero_delta_for_identical(self):
        base, taught, delta = stage_evaluate("same text", "same text")
        assert delta == 0.0

    def test_stage_distill_zero_delta_returns_none(self):
        cache = ReflexCache()
        rid = stage_distill("topic", "sit", "lesson", cache, delta=0.0)
        assert rid is None
        assert cache.size() == 0

    def test_stage_distill_high_delta_caps_confidence(self):
        cache = ReflexCache()
        rid = stage_distill("topic", "sit", "lesson", cache, delta=10.0)
        reflex = cache._by_id[rid]
        assert reflex.confidence == 0.95  # capped
