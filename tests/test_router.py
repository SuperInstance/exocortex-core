import numpy as np
import pytest

from exocortex.router import ExoRouter, RouteTarget


def _vec(value: float = 0.0, dim: int = 384) -> list[float]:
    return [value] * dim


def test_empty_router_routes_to_cloud():
    router = ExoRouter()
    decision = router.route(_vec(0.0))
    assert decision.target == RouteTarget.CLOUD
    assert decision.confidence == pytest.approx(0.0)


def test_positive_local_outcome_moves_boundary():
    router = ExoRouter()
    emb = _vec(0.5)

    # First, unknown → CLOUD.
    decision = router.route(emb)
    assert decision.target == RouteTarget.CLOUD

    # Record a solid local outcome.
    router.record_outcome(emb, quality=0.7)

    # Re-route the same embedding; it should now be LOCAL.
    decision = router.route(emb)
    assert decision.target == RouteTarget.LOCAL
    assert decision.confidence > 0.5


def test_thick_fog_overrides_to_cloud():
    router = ExoRouter(fog_threshold=0.01)
    emb = _vec(0.5)
    # Record an outcome far away in embedding space so fog is huge.
    far_emb = _vec(0.99)
    router.record_outcome(far_emb, quality=0.9)

    decision = router.route(emb)
    assert decision.target == RouteTarget.CLOUD
    assert "fog" in decision.reason.lower()


def test_state_dict_roundtrip():
    router = ExoRouter()
    emb = _vec(0.1)
    router.record_outcome(emb, quality=0.8, metadata={"task": "test"})

    state = router.state_dict()
    restored = ExoRouter.from_state_dict(state)
    assert restored.route(emb).target == RouteTarget.LOCAL
