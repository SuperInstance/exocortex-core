"""Batten-spline cascade routing for REFLEX / LOCAL / CLOUD."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from batten_spline import CascadeRouter as BSCascadeRouter


class RouteTarget(str, Enum):
    """Cascade routing targets."""

    REFLEX = "REFLEX"
    LOCAL = "LOCAL"
    CLOUD = "CLOUD"


@dataclass(frozen=True)
class RouteDecision:
    """A routing verdict with explainability signals."""

    target: RouteTarget
    confidence: float
    fog_density: float
    reason: str
    signals: Dict[str, float] = field(default_factory=dict)


class ExoRouter:
    """Model-agnostic cascade router built on ``batten-spline``.

    The router treats the local/cloud boundary as a confidence surface over
    embedding space. Outcomes are written back as new battens so the boundary
    moves over time — more territory becomes LOCAL as the exocortex learns.
    """

    DEFAULT_TARGETS: Dict[str, Dict[str, Any]] = {
        "REFLEX": {
            "threshold": 0.90,
            "description": "high-confidence reflex territory",
        },
        "LOCAL": {
            "threshold": 0.55,
            "description": "local model is reliable in this neighbourhood",
        },
        "CLOUD": {
            "threshold": 0.0,
            "description": "unfamiliar territory; escalate to cloud",
        },
    }

    def __init__(
        self,
        cascade_router: Optional[BSCascadeRouter] = None,
        fog_threshold: float = 1.5,
    ) -> None:
        self.cascade = cascade_router or BSCascadeRouter(
            targets=dict(self.DEFAULT_TARGETS)
        )
        self.fog_threshold = fog_threshold

    def route(self, embedding: List[float] | np.ndarray) -> RouteDecision:
        """Route a prompt embedding to REFLEX, LOCAL, or CLOUD."""
        arr = np.asarray(embedding, dtype=float)
        result = self.cascade.route(arr)

        target = RouteTarget(result.target)
        reason = result.reason

        # Override with cloud when fog is too thick, unless we already have a
        # reflex-level hit.
        if target != RouteTarget.REFLEX and result.fog_density > self.fog_threshold:
            target = RouteTarget.CLOUD
            reason = f"{reason}; overridden to CLOUD by thick fog"

        return RouteDecision(
            target=target,
            confidence=round(float(result.confidence), 6),
            fog_density=round(float(result.fog_density), 6),
            reason=reason,
            signals={
                "raw_confidence": round(float(result.confidence), 6),
                "fog_density": round(float(result.fog_density), 6),
            },
        )

    def record_outcome(
        self,
        embedding: List[float] | np.ndarray,
        quality: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Feed an observed outcome back into the spline."""
        arr = np.asarray(embedding, dtype=float)
        self.cascade.report_outcome(arr, quality, metadata=metadata or {})

    def state_dict(self) -> Dict[str, Any]:
        return self.cascade.state_dict()

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "ExoRouter":
        cascade = BSCascadeRouter.from_state_dict(state)
        return cls(cascade_router=cascade)
