"""Voice gate — STT pattern matching as the first cascade gate."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RouteTarget(str, Enum):
    """Cascade target produced by the voice gate."""

    REFLEX = "REFLEX"
    LOCAL = "LOCAL"
    CLOUD = "CLOUD"


@dataclass(frozen=True)
class Trigger:
    """A pre-approved voice trigger."""

    category: str
    phrase: str
    action: str
    match_type: str = "substring"  # "exact" | "substring"


@dataclass(frozen=True)
class VoiceDecision:
    """Result of voice-gate classification."""

    target: RouteTarget
    confidence: float
    urgent: bool
    trigger: Optional[Trigger] = None
    matched_phrase: Optional[str] = None
    reason: str = ""


# Urgency signals that can override normal routing.
URGENCY_WORDS = {"stop", "halt", "wait", "now"}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _target_for_action(action: str) -> RouteTarget:
    """Infer cascade target from the trigger action string."""
    lowered = action.lower()
    if lowered.startswith("reflex"):
        return RouteTarget.REFLEX
    if lowered.startswith("local"):
        return RouteTarget.LOCAL
    if lowered.startswith("cloud"):
        return RouteTarget.CLOUD
    # Default to local for safety — the downstream router can still escalate.
    return RouteTarget.LOCAL


class VoiceGate:
    """Pattern-match raw STT text into cascade decisions.

    The gate never emits an executable command; it emits only intent phrases
    mapped to known actions. Urgency signals ("stop", "halt", "wait", "now")
    flag decisions for a hard veto path.
    """

    def __init__(self, triggers: Optional[List[Trigger]] = None) -> None:
        self._triggers: List[Trigger] = list(triggers or [])

    def register(self, trigger: Trigger) -> None:
        self._triggers.append(trigger)

    def classify(self, transcript: str) -> VoiceDecision:
        """Classify a raw STT transcript."""
        normalized = _normalize(transcript)
        urgent = self._detect_urgency(normalized)

        # 1) Exact match
        exact = self._match_exact(normalized)
        if exact is not None:
            trigger, matched = exact
            return VoiceDecision(
                target=_target_for_action(trigger.action),
                confidence=1.0,
                urgent=urgent,
                trigger=trigger,
                matched_phrase=matched,
                reason=f"Exact voice trigger '{trigger.phrase}' ({trigger.category})",
            )

        # 2) Substring match
        substring = self._match_substring(normalized)
        if substring is not None:
            trigger, matched = substring
            return VoiceDecision(
                target=_target_for_action(trigger.action),
                confidence=0.85,
                urgent=urgent,
                trigger=trigger,
                matched_phrase=matched,
                reason=f"Substring voice trigger '{trigger.phrase}' ({trigger.category})",
            )

        # 3) Unknown / ambiguous
        return VoiceDecision(
            target=RouteTarget.CLOUD,
            confidence=0.2,
            urgent=urgent,
            reason="No pre-approved voice trigger matched",
        )

    def _detect_urgency(self, normalized: str) -> bool:
        words = set(normalized.split())
        return not words.isdisjoint(URGENCY_WORDS)

    def _match_exact(
        self, normalized: str
    ) -> Optional[tuple[Trigger, str]]:
        for trigger in self._triggers:
            if trigger.match_type == "exact" and trigger.phrase.lower() == normalized:
                return trigger, trigger.phrase
        return None

    def _match_substring(
        self, normalized: str
    ) -> Optional[tuple[Trigger, str]]:
        for trigger in self._triggers:
            if trigger.match_type == "substring" and trigger.phrase.lower() in normalized:
                return trigger, trigger.phrase
        return None

    def list_triggers(self) -> List[Trigger]:
        return list(self._triggers)
