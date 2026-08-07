"""SuperInstance Exocortex — external brain architecture for small local models."""

from __future__ import annotations

from .bond import (
    BOND_POINTS,
    BondEvent,
    BondGate,
    TIER_NAMES,
    TIER_THRESHOLDS,
    award_bond,
    tier_for,
    tier_name,
)
from .distiller import (
    run_iteration,
    stage_distill,
    stage_evaluate,
    stage_student,
    stage_teacher,
    stage_update_prompt,
)
from ._embed import hash_embedding
from .memory import InMemoryBackend, MemoryIndex, SQLiteVecBackend, VectorBackend
from .reflex_cache import NailReflex, ReflexCache
from .router import ExoRouter, RouteDecision, RouteTarget
from .voice_gate import Trigger, VoiceDecision, VoiceGate

__version__ = "0.1.0"

__all__ = [
    # Reflex cache
    "NailReflex",
    "ReflexCache",
    # Voice gate
    "Trigger",
    "VoiceDecision",
    "VoiceGate",
    # Memory
    "VectorBackend",
    "InMemoryBackend",
    "SQLiteVecBackend",
    "MemoryIndex",
    # Router
    "RouteTarget",
    "RouteDecision",
    "ExoRouter",
    # Bond
    "BondEvent",
    "BOND_POINTS",
    "TIER_THRESHOLDS",
    "TIER_NAMES",
    "award_bond",
    "tier_for",
    "tier_name",
    "BondGate",
    # Distiller
    "stage_teacher",
    "stage_student",
    "stage_evaluate",
    "stage_distill",
    "stage_update_prompt",
    "run_iteration",
    # Embedding
    "hash_embedding",
    # Meta
    "__version__",
]
