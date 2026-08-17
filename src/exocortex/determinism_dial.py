"""Determinism dial — the cortex feels how deterministic it's being.

The exocortex's embedding machinery is deterministic by construction:
``hash_embedding`` turns the same text into the same vector, every time,
on every platform. That property is normally a *means* — it keeps the
memory index stable so the brain outside the model can be trusted. This
dial turns it into a *sense*.

The elephant's signal-chain thesis
(``elephant/docs/signal-chain-thesis.md``) says a room's signal is not
only *what* is being said but *who or what* is generating it — model
prose at one end, deterministic code at the other — and the elephant
reads that ratio with the ``model_vs_code`` dial. This is the cortex's
answer, one generation newer: not a lexicon bolted onto the room, but
the cortex reading the room *with its own eyes*. Every message gets a
hash-embedding from the cortex's own machinery; a message whose vector
has been seen before is the room repeating itself, and repetition is
determinism. A room that always says something new is generative.

The reading blends two senses:

- the **lexicon bridge** — model_vs_code's words, phrases, and symbols,
  matured into the cortex idiom — scores each message's *nature*:
  code/log/symbol-shaped text is deterministic, prose/hedged/first-person
  text is generative, mapped onto ``[0 deterministic .. 1 generative]``;
- the **embedding statistic** — the genuinely cortical part — measures
  ``repeat_frac``: the fraction of messages whose hash-embedding appears
  more than once in the room. Repetition is deterministic *evidence*;
  novelty alone is not evidence of generativity (a fresh commit is novel
  and still deterministic), so the embedding never pushes the reading up —
  it only attenuates it.

``reading = nature * (1 - repeat_frac)`` — the room's nature, dampened by
how much it is a loop.

This is the 10th dial. The elephant's other nine senses feel the room's
temperature; this one feels the room's *fingerprint* — whether the room
is a loop or a conversation.
"""
from __future__ import annotations

import importlib.util
import math
import os
import re
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ._embed import hash_embedding

# --------------------------------------------------------------------------- #
# Elephant seam — the 10th dial joins the herd.                                #
#                                                                              #
# The Dial ABC lives in the elephant (`elephant/dial.py`). If the elephant is  #
# importable — installed, on ELEPHANT_ROOT, or as a `../elephant` sibling —    #
# DeterminismDial subclasses the real ABC and drops straight into a DialBank.  #
# Otherwise it degrades to a pure duck-typed dial (anything with a `.messages` #
# list of `.text`-bearing objects): no dependency, no failure. The seam never  #
# imports the elephant's heavy modules (field.py pulls numpy); if the whole    #
# package is importable we use it, else we surgically load just `dial.py` +    #
# `room.py`, which are stdlib-only.                                            #
# --------------------------------------------------------------------------- #


def _resolve_elephant_root() -> Optional[str]:
    """Find an elephant source tree: ELEPHANT_ROOT, or a `../elephant` sibling."""
    env = os.environ.get("ELEPHANT_ROOT")
    if env:
        return env
    here = Path(__file__).resolve()
    sibling = here.parents[2].parent / "elephant"  # projects/exocortex-core/../elephant
    if (sibling / "elephant" / "dial.py").is_file():
        return str(sibling)
    return None


def _surgical_elephant(root: str) -> Tuple[type, type]:
    """Load just `elephant/dial.py` + `elephant/room.py` (stdlib-only) so the
    real ABC is available even when the whole package is not importable."""
    pkg_dir = Path(root) / "elephant"
    pkg = types.ModuleType("elephant")
    pkg.__path__ = [str(pkg_dir)]
    pkg.__package__ = "elephant"
    sys.modules["elephant"] = pkg

    def _load(name: str):
        spec = importlib.util.spec_from_file_location(f"elephant.{name}", pkg_dir / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"elephant.{name}"] = mod
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod

    _load("room")
    dial = _load("dial")
    return dial.Dial, dial.DialBank


def _import_elephant() -> Tuple[type, Optional[type]]:
    """Return ``(DialBase, DialBank)`` from the elephant, or ``(object, None)``.

    ``sys.path`` is left exactly as found: a discovered ELEPHANT_ROOT is
    added only for the import attempt and removed afterwards (the loaded
    package, if any, stays cached in ``sys.modules``).
    """
    # 1. Already importable (installed, or on sys.path).
    try:
        import elephant.dial as _dial  # type: ignore

        return _dial.Dial, _dial.DialBank
    except Exception:
        pass
    # 2. Source tree found: try the real package, then the surgical load.
    root = _resolve_elephant_root()
    added = False
    if root and str(root) not in sys.path:
        sys.path.insert(0, str(root))
        added = True
    try:
        if root:
            try:
                import elephant.dial as _dial  # type: ignore

                return _dial.Dial, _dial.DialBank
            except Exception:
                sys.modules.pop("elephant", None)
                sys.modules.pop("elephant.room", None)
                sys.modules.pop("elephant.dial", None)
                try:
                    return _surgical_elephant(root)
                except Exception:
                    sys.modules.pop("elephant", None)
                    sys.modules.pop("elephant.room", None)
                    sys.modules.pop("elephant.dial", None)
        return object, None  # type: ignore
    finally:
        if added:
            sys.path.remove(str(root))


DialBase, DialBank = _import_elephant()
ELEPHANT_AVAILABLE = DialBase is not object

# --------------------------------------------------------------------------- #
# The lexicon bridge — model_vs_code's lexicons, matured into the cortex idiom #
# (ported from elephant/dials/model_vs_code.py so the seam stays optional).    #
# --------------------------------------------------------------------------- #

# Generative: hedges, reflection, first-person, prose, creativity, warmth.
MODEL_WORDS = {
    "i", "we", "my", "our", "me", "us", "you", "your",
    "maybe", "perhaps", "probably", "likely", "arguably", "possibly",
    "feel", "felt", "feels", "feeling", "think", "thinks", "believe",
    "wonder", "wondered", "imagine", "remember", "remembers", "sense",
    "seemed", "seems", "seem", "however", "moreover", "therefore", "thus",
    "indeed", "ultimately", "meanwhile", "furthermore", "nevertheless",
    "story", "voice", "warm", "warmth", "light", "gentle", "soft", "alive",
    "holds", "held", "together", "kind", "wonderful", "beautiful",
    "something", "someone", "everything", "nothing", "ourselves", "myself",
}
MODEL_PHRASES = {
    "i think", "i believe", "i wonder", "i feel", "it seems", "in a sense",
    "sort of", "kind of", "as if", "what if", "to me", "for me",
    "maybe we", "perhaps the", "we are", "we were",
}

# Deterministic: keywords, determinism, diffs, errors, commit discipline —
# plus the cortex's own hash-stable vocabulary (pipeline, cache, relay...).
CODE_WORDS = {
    "def", "fn", "function", "return", "import", "class", "struct",
    "impl", "let", "const", "var", "pub", "match", "enum", "trait",
    "elif", "else", "loop", "while", "typeof", "interface", "namespace",
    "static", "void", "mut", "traceback", "error", "exception", "assert",
    "undefined", "nan", "null", "none", "todo", "fixme", "hack",
    "deprecated", "refactor", "merge", "commit", "push", "rebase", "pull",
    "diff", "patch", "lint", "typecheck", "coverage", "dockerfile",
    "pipeline", "syntaxerror", "keyerror", "typeerror",
    # cortex idiom — words the brain outside the model prints when it runs
    "worker", "relay", "cron", "deploy", "schema", "cache", "index",
    "vector", "embed", "query", "hash", "sql", "api", "config", "payload",
    "endpoint", "queue", "retry", "status", "debug", "log", "logs",
}
CODE_PHRASES = {
    "feat:", "fix:", "chore:", "docs:", "refactor:", "test:", "perf:",
    "build:", "ci:", "revert:", "style:", "release:", "merge ", "commit ",
    "push ", "pull request", "diff --git", "+++ b/", "--- a/", "@@ -",
    "at line", "syntax error", "merge conflict", "type error",
    "null pointer", "undefined behavior", "running tests",
}
# Symbols that read as code: braces, brackets, parens, semicolons, operators.
CODE_SYMBOLS = re.compile(
    r"[{}()\[\];]|->|=>|::|==|!=|<=|>=|\+=|-=|\*=|/=|&&|\|\|"
)
# Log shapes: timestamps, levels, hex addresses, versions, file paths —
# the exact-reproduction markers of a machine writing to the room.
LOG_SHAPES = re.compile(
    r"\d{4}-\d{2}-\d{2}"                       # ISO date
    r"|\d{1,2}:\d{2}(?::\d{2})?"               # clock time
    r"|\[(?:info|warn|warning|error|debug|trace|fatal)\]"
    r"|\bat line \d+"
    r"|0x[0-9a-f]{4,}"                         # hex address
    r"|\b(?:syntaxerror|keyerror|typeerror|indexerror|valueerror|"
    r"assertionerror|importerror)\b"
    r"|\b\d+\.\d+\.\d+\b"                      # version number
    r"|(?:^|\s)[\w./-]+\.(?:py|rs|ts|js|go|java|sh|toml|yaml|yml|json|lock)\b",
    re.IGNORECASE,
)

WORD_RE = re.compile(r"\w+")


def _score_message(text: str) -> float:
    """Score one message's nature, ``[0 deterministic .. 1 generative]``.

    Counts deterministic markers (code words, phrases, symbols, log shapes)
    against generative markers (model words, phrases); 0.5 when the message
    carries no signal either way.
    """
    if not text:
        return 0.5
    lower = text.lower()
    words = set(WORD_RE.findall(lower))
    det = sum(1 for w in words if w in CODE_WORDS)
    det += sum(1 for p in CODE_PHRASES if p in lower)
    det += len(CODE_SYMBOLS.findall(text))
    det += len(LOG_SHAPES.findall(lower))
    gen = sum(1 for w in words if w in MODEL_WORDS)
    gen += sum(1 for p in MODEL_PHRASES if p in lower)
    if det + gen == 0:
        return 0.5
    return gen / (det + gen)


def _text_of(message) -> str:
    return getattr(message, "text", "") or ""


def _fingerprint(vector: List[float], bits: int = 12) -> int:
    """Coarse sign-bit bucket for a vector — the embedding's postal code.

    Only the first ``bits`` dimensions are used: this is a *coarse* bucket
    for entropy, not a similarity measure — two vectors differing past
    ``bits`` may share a bucket, which is exactly what makes the stream
    entropy robust to tiny one-dimensional jitter.
    """
    fp = 0
    for v in vector[:bits]:
        fp = (fp << 1) | (1 if v > 0 else 0)
    return fp


def _entropy(counts: List[int], n: int) -> float:
    """Shannon entropy of the embedding stream, normalized to [0, 1]."""
    if n < 2:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / n
        h -= p * math.log2(p)
    return h / math.log2(min(n, 256))


class _RoomView:
    """A slice of a room — name + messages, all a dial needs to read."""

    def __init__(self, room, messages: List) -> None:
        self.name = getattr(room, "name", "window")
        self.messages = messages


class DeterminismDial(DialBase):
    """The 10th dial: how deterministic vs generative the room's output is.

    Reads a Room (duck-typed: anything with ``.messages`` of ``.text``-bearing
    objects). Each message is hash-embedded with the cortex's own machinery;
    the fraction of repeated vectors is the room's determinism evidence, and
    the model_vs_code lexicon bridge supplies the room's *nature*.
    """

    name = "determinism"
    description = (
        "how deterministic vs generative the room's output is, "
        "[0 deterministic .. 1 generative]"
    )

    def __init__(
        self,
        dim: int = 384,
        embed_fn: Callable[[str, int], List[float]] = hash_embedding,
    ) -> None:
        if dim <= 0 or dim % 4 != 0:
            raise ValueError("dim must be a positive multiple of 4")
        self.dim = dim
        self._embed = embed_fn

    # ------------------------------------------------------------------ #
    # The reading                                                         #
    # ------------------------------------------------------------------ #
    def read(self, room) -> float:
        """Current determinism reading, ``[0 deterministic .. 1 generative]``.

        An empty room has no fingerprint yet — read 0.5, neutral.
        """
        messages = [m for m in (getattr(room, "messages", None) or []) if _text_of(m)]
        if not messages:
            return 0.5
        nature = sum(_score_message(_text_of(m)) for m in messages) / len(messages)
        repeat_frac = self.embedding_stats(room)["repeat_frac"]
        reading = nature * (1.0 - repeat_frac)
        return max(0.0, min(1.0, reading))

    def series(self, room, window: int = 8) -> List[float]:
        """Windowed determinism readings so the dial can be trained over time.

        A window larger than the room collapses to a single reading of the
        whole room; an empty room reads ``[0.5]``.
        """
        messages = list(getattr(room, "messages", None) or [])
        if not messages:
            return [self.read(room)]
        return [
            self.read(_RoomView(room, messages[i : i + window]))
            for i in range(0, len(messages), window)
        ]

    # ------------------------------------------------------------------ #
    # The cortex reading its own embedding stream                         #
    # ------------------------------------------------------------------ #
    def embedding_stats(self, room) -> Dict[str, float]:
        """Statistics over the room's hash-embedding stream.

        - ``repeat_frac``: fraction of messages whose hash-embedding appears
          more than once in the room (the room repeating itself — the room
          is a field, so repetition counts anywhere in it, not just back-to-
          back; a message repeated at message 3 and again at message 40 is
          still the room returning to the same beat).
        - ``novel_frac``: fraction of messages whose vector appears once.
        - ``distinct_vectors``: unique hash-embeddings in the room.
        - ``stream_entropy``: Shannon entropy of the sign-bit fingerprint
          stream, normalized to ``[0, 1]`` — 0.0 for a stuck loop, ~1.0 for
          a room always saying something new.
        """
        messages = [m for m in (getattr(room, "messages", None) or []) if _text_of(m)]
        n = len(messages)
        if n == 0:
            return {
                "messages": 0,
                "repeat_frac": 0.0,
                "novel_frac": 0.0,
                "distinct_vectors": 0,
                "stream_entropy": 0.0,
            }
        vectors = [tuple(self._embed(_text_of(m), self.dim)) for m in messages]
        counts = Counter(vectors)
        repeats = sum(1 for v in vectors if counts[v] > 1)
        buckets = Counter(_fingerprint(v) for v in vectors)
        return {
            "messages": n,
            "repeat_frac": repeats / n,
            "novel_frac": 1.0 - repeats / n,
            "distinct_vectors": len(counts),
            "stream_entropy": _entropy(list(buckets.values()), n),
        }

    def __repr__(self) -> str:
        return f"<DeterminismDial {self.name} [{self.dim}d]>"


def register(bank) -> DeterminismDial:
    """Add the determinism dial to an elephant DialBank (the 10th sense).

    ``bank`` is anything with an ``add(dial)`` method (the elephant's
    ``DialBank``). Returns the dial so callers can read it directly too.
    """
    dial = DeterminismDial()
    if bank is not None and hasattr(bank, "add"):
        bank.add(dial)
    return dial
