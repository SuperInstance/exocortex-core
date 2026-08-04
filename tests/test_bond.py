import pytest

from exocortex.bond import (
    BOND_POINTS,
    BondGate,
    TIER_THRESHOLDS,
    award_bond,
    tier_for,
)


def test_tier_for_boundaries():
    assert tier_for(0) == 0
    assert tier_for(9) == 0
    assert tier_for(10) == 1
    assert tier_for(29) == 1
    assert tier_for(30) == 2
    assert tier_for(69) == 2
    assert tier_for(70) == 3
    assert tier_for(149) == 3
    assert tier_for(150) == 4


def test_award_bond_positive_increases():
    new = award_bond(0, "finished_hook", {})
    assert new == BOND_POINTS["finished_hook"]


def test_award_bond_negative_respects_floor():
    # At tier 1 floor=10, a negative event should not drop below 10.
    new = award_bond(11, "blind_delete", {})
    assert new == 10


def test_award_bond_diminishing_returns():
    events = {"manual_build": 2}
    # Third manual build is halved: 3 // 2 == 1.
    new = award_bond(0, "manual_build", events)
    assert new == 1


def test_award_bond_session_cap():
    events = {"manual_build": 10}  # already earned capped positive points
    # With a low cap the next positive event contributes nothing.
    new = award_bond(30, "manual_build", events, session_cap=5)
    assert new == 30


def test_bond_gate_allows_known_actions():
    gate = BondGate(bond_level=0)
    assert gate.allowed("suggest") is True
    assert gate.allowed("execute_reversible") is False

    gate.update_bond_level(35)
    assert gate.allowed("execute_reversible") is True
    assert gate.allowed("initiate_plan") is False


def test_bond_gate_unknown_action_defaults_to_highest_tier():
    gate = BondGate(bond_level=0)
    assert gate.allowed("unknown_action") is False

    gate.update_bond_level(200)
    assert gate.allowed("unknown_action") is True


def test_bond_gate_ensure_allowed_raises():
    gate = BondGate(bond_level=0)
    with pytest.raises(PermissionError):
        gate.ensure_allowed("execute_reversible")
