"""Bond-level gating for autonomous behaviour."""

from __future__ import annotations

from typing import Dict, Literal, Optional, Set

BondEvent = Literal[
    "session_first_build",
    "finished_hook",
    "manual_build",
    "modify_not_replace",
    "won_argument",
    "returned",
    "blind_delete",
    "ignored_safety_warning",
    "reverted_autonomous_action",
]

# Base point values from the Lucineer scale.
BOND_POINTS: Dict[BondEvent, int] = {
    "session_first_build": 1,
    "finished_hook": 5,
    "manual_build": 3,
    "modify_not_replace": 2,
    "won_argument": 4,
    "returned": 2,
    "blind_delete": -1,
    "ignored_safety_warning": -2,
    "reverted_autonomous_action": -3,
}

# Tier thresholds (Lucineer scale).
TIER_THRESHOLDS = (0, 10, 30, 70, 150)
TIER_NAMES = ("Hired", "Working Together", "Trusted", "Crew", "The Yard")

# Events that represent the core collaboration loop and are exempt from the
# per-session positive cap.
CAP_EXEMPT_EVENTS: Set[BondEvent] = {"finished_hook"}


def tier_for(bond_level: int) -> int:
    """Map a raw bond score to a tier index 0-4."""
    tier = 0
    for i, threshold in enumerate(TIER_THRESHOLDS):
        if bond_level >= threshold:
            tier = i
    return tier


def tier_name(bond_level: int) -> str:
    """Return the human-readable tier name for a bond level."""
    return TIER_NAMES[tier_for(bond_level)]


def _event_value(event: BondEvent, times_this_session: int) -> int:
    """Apply diminishing returns after the 3rd occurrence of an event."""
    base = BOND_POINTS[event]
    if times_this_session < 2:
        return base
    # Halve (rounding toward zero) for repeated grinding.
    return base // 2 if base > 0 else -((-base) // 2)


def _tier_floor(bond_level: int) -> int:
    """The minimum bond level allowed for the current tier."""
    return TIER_THRESHOLDS[tier_for(bond_level)]


def award_bond(
    current_bond: int,
    event: BondEvent,
    events_this_session: Dict[BondEvent, int],
    session_cap: int = 25,
) -> int:
    """Return the new bond level after ``event``.

    Scoring policy:
      - Base values from ``BOND_POINTS``.
      - Diminishing returns after the 3rd occurrence of the same event.
      - Per-session positive cap (default 25), except ``finished_hook``.
      - Bond never drops below the floor of the current tier.

    Args:
        current_bond: current bond score.
        event: which event just fired.
        events_this_session: count of each event already fired this session.
        session_cap: maximum positive bond that can be earned per session.

    Returns:
        New bond level.
    """
    times = events_this_session.get(event, 0)
    delta = _event_value(event, times)

    if delta > 0 and event not in CAP_EXEMPT_EVENTS:
        # Clamp positive delta to the remaining session budget (computed from
        # raw base points so the cap dominates diminishing returns).
        positive_raw_so_far = sum(
            count * BOND_POINTS[ev]
            for ev, count in events_this_session.items()
            if BOND_POINTS[ev] > 0 and ev not in CAP_EXEMPT_EVENTS
        )
        remaining = max(0, session_cap - positive_raw_so_far)
        delta = min(delta, remaining)

    new_bond = current_bond + delta
    floor = _tier_floor(current_bond)
    return max(floor, int(new_bond))


class BondGate:
    """Gate autonomous actions by bond tier."""

    DEFAULT_ACTIONS: Dict[str, int] = {
        "suggest": 0,
        "execute_reversible": 1,
        "execute_medium_risk": 2,
        "initiate_plan": 3,
        "autonomous_background": 4,
    }

    def __init__(
        self,
        bond_level: int = 0,
        action_tiers: Optional[Dict[str, int]] = None,
    ) -> None:
        self.bond_level = bond_level
        self.action_tiers: Dict[str, int] = dict(action_tiers or self.DEFAULT_ACTIONS)

    @property
    def tier(self) -> int:
        return tier_for(self.bond_level)

    def register_action(self, action: str, required_tier: int) -> None:
        """Add an autonomous action and the tier required to perform it."""
        self.action_tiers[action] = required_tier

    def allowed(self, action: str) -> bool:
        """Return ``True`` if the current tier can perform ``action``."""
        required = self.action_tiers.get(action)
        if required is None:
            # Unknown actions default to the highest tier = safest.
            return self.tier >= max(self.action_tiers.values(), default=4)
        return self.tier >= required

    def required_tier(self, action: str) -> int:
        """Return the tier required for ``action``."""
        return self.action_tiers.get(action, max(self.action_tiers.values(), default=4))

    def ensure_allowed(self, action: str) -> None:
        """Raise if the action is not permitted at the current tier."""
        if not self.allowed(action):
            raise PermissionError(
                f"action {action!r} requires tier {self.required_tier(action)}, "
                f"current tier is {self.tier}"
            )

    def update_bond_level(self, bond_level: int) -> None:
        self.bond_level = bond_level
