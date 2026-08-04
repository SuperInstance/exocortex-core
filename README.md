# SuperInstance Exocortex — Core Prototype

The **exocortex** is an external brain that wraps around any small local model.
The local model stays frozen; the brain around it grows through compiled
reflexes, semantic memory, cloud-teacher distillation, cognitive routing, bond
gating, and voice as the first cascade gate.

> The model is small; the brain outside it is not.

## Modules

| Module | Responsibility |
|--------|----------------|
| `reflex_cache` | `.nail` reflex cache with exact-hash and vector-nearest-neighbour lookup, asymmetric confidence updates, and an escape-hatch counter. |
| `voice_gate` | STT pattern matching against a pre-approved trigger table; exact/substring matching, urgency detection, and cascade routing. |
| `memory` | Pluggable semantic-memory index (sqlite-vec when available, in-memory fallback), hash embeddings, upsert/query/delete, and outcome write-back. |
| `router` | `batten-spline` cascade router mapping embeddings to `REFLEX`, `LOCAL`, or `CLOUD` targets; outcome feedback reshapes the spline. |
| `bond` | Lucineer bond scoring, tier thresholds, and a `BondGate` that unlocks autonomous actions as trust accumulates. |
| `distiller` | Teacher → student → compile loop: cloud teacher produces a lesson, local student evaluates it, and positive deltas become `.nail` reflexes. |

## Quick start

```python
from exocortex import ReflexCache, ExoRouter, MemoryIndex

cache = ReflexCache()
cache.store("what is 2+2", "4", confidence=0.9)
print(cache.lookup("what is 2+2").response)  # 4

router = ExoRouter()
# Before learning, unknown embeddings route to CLOUD.
print(router.route([0.1] * 384).target)

memory = MemoryIndex()
memory.upsert("doc1", "The exocortex writes every outcome back to memory.")
print(memory.query("outcome memory", k=1)[0][1])  # similarity score
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## License

MIT — see `LICENSE`.
