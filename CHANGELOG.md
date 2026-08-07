# Changelog

All notable changes to Exocortex-Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `hash_embedding` now exported from the top-level `exocortex` package.
- `CONTRIBUTING.md` with development setup, style guide, architecture overview, and PR checklist.
- 30+ additional tests covering uncovered code paths (vector NN escape-hatch blocking, cloud-action voice triggers, empty-cache persistence, confidence rounding, pipeline integration).
- `tests/test_coverage_gaps.py` — targeted tests for edge cases that were previously uncovered.

### Changed
- `__init__.py` now exports `hash_embedding` in `__all__`.

## [0.1.0] - 2025-08-04

### Added
- **ReflexCache** — `.nail` reflex cache with exact-hash and vector-nearest-neighbour lookup.
  - Asymmetric confidence updates (Pincher's rule: +5% on success, −10% on failure).
  - Escape-hatch counter (`max_consecutive_uses`) prevents high-confidence blind spots.
  - Cosine similarity vector search with configurable threshold.
  - Save/load to `.nail` directory or single JSON file.
- **VoiceGate** — STT pattern matching as the first cascade gate.
  - Exact and substring trigger matching.
  - Urgency detection ("stop", "halt", "wait", "now").
  - Cascade routing to REFLEX / LOCAL / CLOUD targets.
- **MemoryIndex** — Pluggable semantic-memory index.
  - `InMemoryBackend` for zero-dependency operation.
  - `SQLiteVecBackend` for persistent local storage via sqlite-vec.
  - Outcome write-back for the cognitive feedback loop.
- **ExoRouter** — Batten-spline cascade router.
  - Embedding → REFLEX/LOCAL/CLOUD routing with confidence surface.
  - Fog density detection overrides to CLOUD when uncertain.
  - Outcome feedback reshapes the spline boundary.
  - State dict serialization for persistence.
- **BondGate** — Lucineer bond-level gating.
  - 5-tier trust system (Hired → The Yard).
  - Per-session positive cap with diminishing returns.
  - Tier-floor protection prevents de-ranking from isolated negative events.
  - Configurable action-tier mappings.
- **Distiller** — Teacher → student → compile distillation loop.
  - Four-axis evaluation (novelty, specificity, engagement, spatial).
  - Positive-delta compilation into `.nail` reflexes.
  - Consecutive-positive threshold for prompt promotion.
  - Full iteration runner with injected teacher/student functions.
- **hash_embedding** — Deterministic SHA-256-based embedding with no external dependencies.
- 36 tests covering all six modules.
