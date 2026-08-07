# Contributing to Exocortex-Core

Thank you for your interest in improving exocortex-core! This document covers
the development workflow, testing expectations, and code style guidelines.

## Development Setup

```bash
git clone https://github.com/SuperInstance/exocortex-core.git
cd exocortex-core
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
# Full suite
pytest

# With coverage report
pytest --cov=exocortex --cov-report=term-missing

# Verbose, specific module
pytest tests/test_reflex_cache.py -v
```

Tests auto-discover anything in `tests/` matching `test_*.py`.

## Coverage Goals

| Module         | Current | Target |
|----------------|---------|--------|
| `_embed.py`    | 100%    | 100%   |
| `bond.py`      | 100%    | 100%   |
| `distiller.py` | 100%    | 100%   |
| `router.py`    | 100%    | 100%   |
| `voice_gate.py`| 99%     | 100%   |
| `reflex_cache.py` | 96%  | 98%+   |
| `memory.py`    | 71%*    | 90%+   |

\* The `SQLiteVecBackend` paths are only exercisable when `sqlite_vec` is
installed. Tests for those paths are skipped automatically when the package
is unavailable. Do not remove them — they run in CI environments where
`sqlite_vec` is present.

## Code Style

- **Python ≥ 3.10** — use `from __future__ import annotations` for forward refs.
- **Type hints** on all public functions and methods.
- **Dataclasses** for structured data (`NailReflex`, `RouteDecision`, etc.).
- **No external dependencies** beyond `numpy` and `batten-spline` in core.
  Optional dependencies (`sqlite_vec`) must degrade gracefully.
- **Docstrings** on every public class and function.

## Architecture

```
src/exocortex/
├── __init__.py        # Public API
├── _embed.py          # Deterministic hash embedding (no deps)
├── reflex_cache.py    # .nail reflex cache (exact + vector NN lookup)
├── voice_gate.py      # STT pattern matching → cascade decision
├── memory.py          # Semantic memory (pluggable vector backends)
├── router.py          # Batten-spline cascade router
├── bond.py            # Lucineer bond scoring + tier gating
└── distiller.py       # Teacher → student → compile distillation
```

### Design Principles

1. **The model stays frozen.** All learning happens in the exocortex layer.
2. **No network calls in core.** External functions are injected (e.g.
   `teacher_fn`, `student_fn`, `embed_fn`).
3. **Graceful degradation.** Every optional dependency must have a working
   fallback.
4. **Explainability.** Every decision carries a `reason` string and signals.

## Adding a New Module

1. Create `src/exocortex/your_module.py` with type hints and docstrings.
2. Export public symbols in `src/exocortex/__init__.py` and add to `__all__`.
3. Write tests in `tests/test_your_module.py`.
4. Update `README.md` with a row in the Modules table.
5. Add a `CHANGELOG.md` entry under `[Unreleased]`.

## Pull Request Checklist

- [ ] All tests pass (`pytest -v`)
- [ ] Coverage does not decrease
- [ ] New public symbols are exported and documented
- [ ] `CHANGELOG.md` updated
- [ ] No secrets or credentials committed

## Reporting Issues

Use [GitHub Issues](https://github.com/SuperInstance/exocortex-core/issues).
Include:
- Python version (`python3 --version`)
- OS
- Steps to reproduce
- Expected vs actual behaviour

## Licence

By contributing, you agree that your contributions are licensed under the MIT
licence (see `LICENSE`).
