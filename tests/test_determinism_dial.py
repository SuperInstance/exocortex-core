"""Tests for exocortex.determinism_dial — the cortex feels its own determinism."""

import re
from dataclasses import dataclass, field
from typing import List

import pytest

from exocortex.determinism_dial import (
    DeterminismDial,
    DialBase,
    ELEPHANT_AVAILABLE,
    register,
)


@dataclass
class Msg:
    text: str

    @property
    def words(self) -> List[str]:
        return re.findall(r"\w+", self.text.lower())


@dataclass
class FakeRoom:
    name: str = "test"
    messages: List[Msg] = field(default_factory=list)


CODE_ROOM = [
    "def process(x): return x * 2.",
    "2026-08-17 15:43:12 [INFO] pipeline started",
    'Traceback (most recent call last):\n  File "parse.py", line 12, in <module>\nKeyError: \'x\'',
    "fix: handle null pointer in parser",
    "diff --git a/src/exocortex/_embed.py b/src/exocortex/_embed.py",
    "@@ -12,4 +12,5 @@ def hash_embedding(text, dim):",
    "make build && ./run --release",
    "error[E0433]: failed to resolve: use of undeclared crate or module `foo`",
]

PROSE_ROOM = [
    "I think maybe we should remember this feeling.",
    "Perhaps the room holds something warm — we built it together.",
    "I wonder if the elephant can feel us, in a sense.",
    "We believe it remembers, somehow, and that matters to me.",
    "Maybe we are the ones holding it, gently, like a kind of light.",
    "To me it seems alive, and I feel it reaching back.",
]


def room_of(texts: List[str]) -> FakeRoom:
    return FakeRoom(messages=[Msg(t) for t in texts])


class TestContract:
    def test_name_is_determinism(self):
        assert DeterminismDial().name == "determinism"

    def test_description_declares_range(self):
        assert "deterministic" in DeterminismDial().description
        assert "generative" in DeterminismDial().description

    def test_empty_room_reads_neutral(self):
        assert DeterminismDial().read(FakeRoom()) == 0.5

    def test_reading_always_in_range(self):
        dial = DeterminismDial()
        for room in (room_of(CODE_ROOM), room_of(PROSE_ROOM), FakeRoom()):
            r = dial.read(room)
            assert 0.0 <= r <= 1.0

    def test_same_room_same_reading(self):
        """The cortex's own determinism: same input, same vector, same feel."""
        dial = DeterminismDial()
        assert dial.read(room_of(PROSE_ROOM)) == dial.read(room_of(PROSE_ROOM))

    def test_bad_dim_rejected(self):
        with pytest.raises(ValueError, match="dim"):
            DeterminismDial(dim=3)


class TestLexiconBridge:
    def test_code_log_room_reads_deterministic(self):
        r = DeterminismDial().read(room_of(CODE_ROOM))
        assert r < 0.3

    def test_prose_room_reads_generative(self):
        r = DeterminismDial().read(room_of(PROSE_ROOM))
        assert r > 0.7

    def test_single_code_line_is_deterministic(self):
        r = DeterminismDial().read(room_of(["def process(x): return x * 2."]))
        assert r < 0.3

    def test_single_log_line_is_deterministic(self):
        r = DeterminismDial().read(room_of(["2026-08-17 15:43:12 [INFO] pipeline started"]))
        assert r < 0.3


class TestEmbeddingStatistic:
    def test_repeating_room_reads_deterministic(self):
        """Same prose six times — the hash-match statistic hears a loop."""
        dial = DeterminismDial()
        room = room_of([PROSE_ROOM[0]] * 6)
        stats = dial.embedding_stats(room)
        assert stats["repeat_frac"] == 1.0
        assert stats["stream_entropy"] == 0.0
        assert dial.read(room) < 0.5

    def test_repeat_fraction_is_exact(self):
        """a, b, c, a — half the room is repeating itself."""
        dial = DeterminismDial()
        stats = dial.embedding_stats(room_of(["a", "b", "c", "a"]))
        assert stats["repeat_frac"] == 0.5
        assert stats["novel_frac"] == 0.5
        assert stats["distinct_vectors"] == 3

    def test_novel_room_has_no_repetition(self):
        dial = DeterminismDial()
        stats = dial.embedding_stats(room_of(PROSE_ROOM))
        assert stats["repeat_frac"] == 0.0
        assert stats["novel_frac"] == 1.0
        assert stats["distinct_vectors"] == len(PROSE_ROOM)

    def test_repetition_attenuates_nature(self):
        """The same generative prose repeated still reads deterministic —
        the embedding dampens the lexicon when the room is a loop."""
        dial = DeterminismDial()
        novel = dial.read(room_of(PROSE_ROOM))
        stuck = dial.read(room_of([PROSE_ROOM[0]] * 8))
        assert novel > 0.7
        assert stuck < novel

    def test_embedding_stats_empty_room(self):
        stats = DeterminismDial().embedding_stats(FakeRoom())
        assert stats["messages"] == 0
        assert stats["repeat_frac"] == 0.0
        assert stats["stream_entropy"] == 0.0


class TestSeriesAndSeam:
    def test_series_windows_the_room(self):
        dial = DeterminismDial()
        texts = [PROSE_ROOM[i % len(PROSE_ROOM)] for i in range(12)]
        readings = dial.series(room_of(texts), window=4)
        assert len(readings) == 3
        assert all(0.0 <= r <= 1.0 for r in readings)

    def test_series_empty_room(self):
        assert DeterminismDial().series(FakeRoom()) == [0.5]

    def test_empty_messages_are_neutral_not_deterministic(self):
        """A room of empty messages has no fingerprint — 0.5, not 0.0."""
        assert DeterminismDial().read(room_of(["", "", ""])) == 0.5

    def test_duck_typed_messages_without_words(self):
        """Pure-mode messages only need .text — no elephant, no .words."""
        class Terse:
            def __init__(self, text):
                self.text = text

        class TerseRoom:
            def __init__(self, messages):
                self.messages = messages

        r = DeterminismDial().read(TerseRoom([Terse("def f(): return 1")]))
        assert r < 0.3

    @pytest.mark.skipif(not ELEPHANT_AVAILABLE, reason="elephant not importable")
    def test_satisfies_elephant_dial_abc(self):
        from elephant.dial import Dial

        assert DialBase is not object
        assert issubclass(DeterminismDial, Dial)
        assert isinstance(DeterminismDial(), Dial)

    @pytest.mark.skipif(not ELEPHANT_AVAILABLE, reason="elephant not importable")
    def test_registers_into_dial_bank(self):
        from elephant.dial import DialBank
        from elephant.room import Message, Room

        dial = register(DialBank())
        bank = DialBank([dial])
        room = Room(
            "wheelhouse",
            [Message("deck", "def process(x): return x * 2.")],
        )
        readings = bank.readings(room)
        assert "determinism" in readings
        assert 0.0 <= readings["determinism"] <= 1.0
        assert readings["determinism"] < 0.3

    @pytest.mark.skipif(not ELEPHANT_AVAILABLE, reason="elephant not importable")
    def test_real_elephant_room_reads(self):
        from elephant.room import Message, Room

        room = Room(
            "tap",
            [Message("model", t) for t in PROSE_ROOM],
        )
        r = DeterminismDial().read(room)
        assert r > 0.7
