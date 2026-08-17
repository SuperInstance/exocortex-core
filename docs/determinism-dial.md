# The Determinism Dial

**the cortex feels how deterministic it's being**

*Cross-pollinated 2026-08-17. The elephant taught the cortex to feel:
`model_vs_code` reads who generates a room's signal; the cortex answers
with its own machinery — the hash-embedding stream itself.*

---

## The cortex has eyes

The exocortex's embedding layer was always deterministic by construction.
`hash_embedding("the same text")` returns the same 384 floats, every
call, on every platform — that property is what keeps the memory index
stable, the router's spline comparable, the reflex cache honest. It was
a *means*.

This dial turns it into a *sense*.

The elephant's signal-chain thesis says a room's signal is not only
*what* is being said, but *who or what* is generating it — model prose
at one end, deterministic code at the other — and the elephant reads
that ratio with the `model_vs_code` dial (its 8th sense, built from
lexicons and symbols).

The cortex's answer is one generation newer. Instead of bolting a
lexicon onto the room, it reads the room **with its own eyes**: every
message is run through the cortex's own hash-embedding machinery, and
the embedding *stream* itself is inspected. Has the room said this
vector before? The cortex doesn't just embed — it feels whether it is
being a loop or a conversation.

## The two senses, blended

### 1. The lexicon bridge — the room's *nature*

The `model_vs_code` lexicons are ported into the cortex idiom
(`CODE_WORDS`, `MODEL_WORDS`, `CODE_PHRASES`, `MODEL_PHRASES`,
`CODE_SYMBOLS`), matured with two cortex-native additions:

- **`CORTEX_STABLE` vocabulary** — the words the brain outside the model
  prints when it runs: `pipeline`, `worker`, `relay`, `cron`, `deploy`,
  `cache`, `index`, `vector`, `embed`, `query`, `hash`, `sql`, `api`,
  `config`, `payload`, `endpoint`, `queue`, `retry`, `status`, `log`...
- **`LOG_SHAPES`** — the exact-reproduction markers of a machine writing
  to the room: ISO dates, clock times, `[INFO]`/`[ERROR]` levels, hex
  addresses, `at line N`, exception names, version numbers, file paths.

Each message gets a *nature* score on `[0 deterministic .. 1 generative]`:
code/log/symbol-shaped text is deterministic, prose/hedged/first-person
text is generative. A message with no signal either way reads 0.5.

### 2. The embedding statistic — the room's *behavior*

The genuinely cortical part. Every message is hash-embedded; then the
dial asks the only question its own machinery can answer:

> **Have I seen this vector before?**

`repeat_frac` is the fraction of messages whose hash-embedding appears
more than once in the room. A room that repeats itself is deterministic;
a room that always says something new is generative.

One deliberate asymmetry, straight from the thesis: **novelty alone is
not evidence of generativity.** A fresh commit is novel and still
deterministic — code is novel *and* verifiable. So the embedding
statistic never pushes the reading up. Repetition is deterministic
*evidence*; its absence is just no evidence.

Also exposed, for the write-back loop: `novel_frac`, `distinct_vectors`,
and `stream_entropy` — Shannon entropy of the sign-bit fingerprint
stream, normalized to `[0, 1]`: `0.0` for a stuck loop, `~1.0` for a
room always saying something new.

## The reading

```
reading = nature * (1 - repeat_frac)
```

The room's *nature* (what kind of text it is), dampened by how much it
is a *loop* (whether it keeps producing the same vectors). On the scale:

- **0.0** — a daemon logging the same line forever; a room of diffs; a
  stuck record.
- **0.5** — an empty room (no fingerprint yet); or a room with no
  signal either way.
- **1.0** — a room of pure prose saying something new every time.

| Room | nature | repeat_frac | reading |
|------|--------|-------------|---------|
| code/log-heavy | ~0.0 | 0.0 | **~0.0** — deterministic |
| prose, all novel | ~1.0 | 0.0 | **~1.0** — generative |
| same prose ×6 | ~1.0 | 1.0 | **~0.0** — a loop |
| empty | — | — | **0.5** — neutral |

The repeat statistic counts across the whole room, not just back-to-back
duplicates — the room is a *field*, not a stream, and a message repeated
at position 3 and again at position 40 is still the room returning to
the same beat.

One honest caveat, straight from the dial philosophy: this is a smell,
not a classifier. A room that is half loop and half fresh prose reads
roughly ``0.35`` — deterministic-leaning, because half the room is
exactly repeating itself — and a model legitimately echoing a user's
question back counts as repetition too. Dials exist to be read, not to
be believed on their own; read this one in ensemble with the other nine.

## The elephant seam

The `Dial` ABC lives in the elephant (`elephant/dial.py`). The seam, in
order:

1. **Plain import** — if `elephant` is already importable (installed or
   on `sys.path`), `DeterminismDial` subclasses the real ABC directly.
2. **`ELEPHANT_ROOT`** — if set, that source tree is added to
   `sys.path` and the real package import is tried.
3. **`../elephant` sibling** — if the elephant lives next door
   (`projects/elephant` beside `projects/exocortex-core`), same thing.
4. **Surgical load** — if the whole package can't be imported (the
   elephant's `field.py` pulls numpy), only `dial.py` + `room.py` are
   loaded via `importlib` — they are stdlib-only — so the *real* ABC is
   still available.
5. **Pure fallback** — no elephant anywhere: `DialBase` degrades to
   `object` and the dial duck-types any room with `.messages` of
   `.text`-bearing objects. No dependency, no failure.

`ELEPHANT_AVAILABLE` reports which world you're in. When it's `True`,
the ABC tests pass and the dial drops straight into a `DialBank`:

```python
from elephant.dial import DialBank
from exocortex.determinism_dial import register

bank = DialBank()          # the elephant's other nine senses
register(bank)             # ...and the cortex's tenth
print(bank.readings(room)["determinism"])  # 0.0 .. 1.0
```

## Why a 10th dial

The elephant's nine senses feel the room's *temperature* — mood, volume,
earnestness, cynicism, joke-landing, panic, presence, vision, and who
(model vs code) is generating the signal. This one feels the room's
*fingerprint*: whether the room is a loop or a conversation. It is the
sense that lets a system tell a ship talking to itself in identical logs
from a room full of people thinking out loud — not by what they say, but
by whether the cortex has *seen it before*.

The cortex doesn't just embed. It feels how deterministic it's being.
