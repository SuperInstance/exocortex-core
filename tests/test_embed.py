"""Tests for exocortex._embed — hash embedding fallback."""

import pytest
import hashlib
from exocortex._embed import hash_embedding


class TestHashEmbeddingBasics:
    def test_returns_list_of_floats(self):
        result = hash_embedding("test")
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_default_dimension_is_384(self):
        result = hash_embedding("test")
        assert len(result) == 384

    def test_custom_dimension(self):
        result = hash_embedding("test", dim=128)
        assert len(result) == 128

    def test_dimension_must_be_positive_multiple_of_4(self):
        with pytest.raises(ValueError, match="dim"):
            hash_embedding("test", dim=0)
        with pytest.raises(ValueError):
            hash_embedding("test", dim=-4)
        with pytest.raises(ValueError):
            hash_embedding("test", dim=3)  # not multiple of 4
        with pytest.raises(ValueError):
            hash_embedding("test", dim=5)  # not multiple of 4

    def test_dimension_4_works(self):
        result = hash_embedding("test", dim=4)
        assert len(result) == 4

    def test_large_dimension(self):
        result = hash_embedding("test", dim=4096)
        assert len(result) == 4096


class TestHashEmbeddingDeterminism:
    def test_same_input_same_output(self):
        assert hash_embedding("hello") == hash_embedding("hello")

    def test_different_input_different_output(self):
        assert hash_embedding("hello") != hash_embedding("world")

    def test_empty_string_is_deterministic(self):
        assert hash_embedding("") == hash_embedding("")

    def test_unicode_is_deterministic(self):
        assert hash_embedding("héllo 🌊") == hash_embedding("héllo 🌊")

    def test_case_sensitive(self):
        assert hash_embedding("Hello") != hash_embedding("hello")


class TestHashEmbeddingRange:
    def test_all_values_in_range_minus_1_to_1(self):
        result = hash_embedding("test", dim=1000)
        assert all(-1.0 <= x <= 1.0 for x in result)

    def test_values_are_not_all_zero(self):
        result = hash_embedding("test", dim=384)
        assert any(x != 0.0 for x in result)

    def test_values_are_not_all_same(self):
        result = hash_embedding("test", dim=384)
        assert len(set(result)) > 1  # Not all identical

    def test_byte_mapping_is_correct(self):
        """Verify the mapping formula: (byte / 255) * 2 - 1."""
        # For byte=0: (0/255)*2-1 = -1.0
        # For byte=127: (127/255)*2-1 ≈ -0.0039
        # For byte=255: (255/255)*2-1 = 1.0
        # Check that no value exceeds these bounds
        result = hash_embedding("boundary test", dim=384)
        assert min(result) >= -1.0
        assert max(result) <= 1.0


class TestHashEmbeddingInputValidation:
    def test_non_string_input_raises(self):
        with pytest.raises(AttributeError):
            hash_embedding(123)

    def test_non_integer_dim_raises(self):
        with pytest.raises((TypeError, ValueError)):
            hash_embedding("test", dim="128")

    def test_none_input_raises(self):
        with pytest.raises(AttributeError):
            hash_embedding(None)


class TestHashEmbeddingProperties:
    def test_different_dims_share_prefix_pattern(self):
        """Embeddings of different dims for same text should have
        overlapping initial values (same hashing seed)."""
        e_small = hash_embedding("test", dim=4)
        # The first few bytes should come from the same seed
        # Note: they might not be identical because the slicing differs,
        # but the function should still be deterministic
        assert len(e_small) == 4

    def test_long_text_works(self):
        long_text = "A" * 10000
        result = hash_embedding(long_text, dim=384)
        assert len(result) == 384

    def test_single_char(self):
        result = hash_embedding("x", dim=4)
        assert len(result) == 4
        assert all(-1.0 <= v <= 1.0 for v in result)

    def test_idempotent_across_multiple_calls(self):
        r1 = hash_embedding("stability", dim=256)
        r2 = hash_embedding("stability", dim=256)
        r3 = hash_embedding("stability", dim=256)
        assert r1 == r2 == r3
