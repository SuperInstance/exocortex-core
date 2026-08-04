"""Teacher → student → compile distillation loop."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .reflex_cache import ReflexCache


TeacherFn = Callable[[str], str]
StudentFn = Callable[[str], str]


@dataclass
class Evaluation:
    """Scores for a single distillation evaluation."""

    novelty: float
    specificity: float
    engagement: float
    spatial: float

    @property
    def average(self) -> float:
        return (self.novelty + self.specificity + self.engagement + self.spatial) / 4.0


def stage_teacher(prompt: str, teacher_fn: TeacherFn) -> str:
    """Run the cloud teacher on a prompt and return the lesson."""
    return teacher_fn(prompt)


def stage_student(prompt: str, student_fn: StudentFn, lesson: Optional[str] = None) -> str:
    """Run the local student, optionally primed with a teacher lesson."""
    if lesson:
        primed = f"[LESSON] {lesson}\n\n[PROMPT] {prompt}"
    else:
        primed = prompt
    return student_fn(primed)


def stage_evaluate(
    baseline: str,
    taught: str,
    topic: Optional[str] = None,
) -> tuple[Evaluation, Evaluation, float]:
    """Score baseline and taught responses on four quality dimensions.

    Returns ``(baseline_scores, taught_scores, delta)`` where ``delta`` is the
    average improvement of the taught response over the baseline.
    """
    base = _score_response(baseline, taught)
    taught_scores = _score_response(taught, baseline)
    delta = taught_scores.average - base.average
    return base, taught_scores, round(delta, 4)


def stage_distill(
    topic: str,
    situation: str,
    lesson: str,
    cache: ReflexCache,
    delta: float,
) -> Optional[str]:
    """Compile a positive-delta lesson into a ``.nail`` reflex.

    Returns the reflex id if the delta was positive, otherwise ``None``.
    """
    if delta <= 0:
        return None

    signature = f"topic={topic} situation={situation}"
    confidence = min(0.95, max(0.05, 0.6 + delta * 0.4))
    return cache.store(
        signature=signature,
        response=lesson,
        confidence=confidence,
        source="cloud:distillation",
        situation=situation,
        metadata={"topic": topic, "delta": round(delta, 4)},
    )


def stage_update_prompt(
    topic: str,
    consecutive_positives: int,
    threshold: int = 3,
) -> Optional[str]:
    """Promote a concise directive after ``threshold`` consecutive positives."""
    if consecutive_positives < threshold:
        return None
    return f"[{topic}] Use the distilled guidance for this topic."


def run_iteration(
    topic: str,
    situation: str,
    prompt: str,
    teacher_fn: TeacherFn,
    student_fn: StudentFn,
    cache: ReflexCache,
    consecutive_positives: int = 0,
    promotion_threshold: int = 3,
) -> Dict[str, Any]:
    """Execute one full teacher→student→compile iteration.

    No network calls are made; ``teacher_fn`` and ``student_fn`` are injected
    callables so tests can run with deterministic stubs.
    """
    lesson = stage_teacher(prompt, teacher_fn)
    baseline = stage_student(prompt, student_fn, lesson=None)
    taught = stage_student(prompt, student_fn, lesson=lesson)

    base_scores, taught_scores, delta = stage_evaluate(baseline, taught, topic=topic)

    reflex_id = stage_distill(topic, situation, lesson, cache, delta)

    if delta > 0:
        consecutive_positives += 1
    else:
        consecutive_positives = 0

    prompt_update = stage_update_prompt(
        topic, consecutive_positives, threshold=promotion_threshold
    )

    return {
        "topic": topic,
        "situation": situation,
        "prompt": prompt,
        "lesson": lesson,
        "baseline": baseline,
        "taught": taught,
        "baseline_scores": {
            "novelty": base_scores.novelty,
            "specificity": base_scores.specificity,
            "engagement": base_scores.engagement,
            "spatial": base_scores.spatial,
            "average": base_scores.average,
        },
        "taught_scores": {
            "novelty": taught_scores.novelty,
            "specificity": taught_scores.specificity,
            "engagement": taught_scores.engagement,
            "spatial": taught_scores.spatial,
            "average": taught_scores.average,
        },
        "delta": delta,
        "reflex_id": reflex_id,
        "consecutive_positives": consecutive_positives,
        "prompt_update": prompt_update,
    }


# ---------------------------------------------------------------------------
# Deterministic scoring helpers
# ---------------------------------------------------------------------------

_SPATIAL_WORDS = {
    "north", "south", "east", "west", "up", "down", "left", "right",
    "above", "below", "beside", "behind", "front", "near", "far",
    "over", "under", "inside", "outside", "between", "around",
}

_SPECIFIC_WORDS = {
    "because", "therefore", "specifically", "exactly", "precisely",
    "measure", "size", "width", "height", "length", "distance",
}


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _unique_bigrams(tokens: List[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:]))


def _score_response(text: str, reference: str) -> Evaluation:
    tokens = _tokens(text)
    ref_tokens = _tokens(reference)

    # Novelty: fraction of bigrams in text that are not in reference.
    text_bigrams = _unique_bigrams(tokens)
    ref_bigrams = _unique_bigrams(ref_tokens)
    novel = text_bigrams - ref_bigrams
    novelty = len(novel) / max(1, len(text_bigrams))

    # Specificity: density of concrete / precise words plus numeric tokens.
    spec_count = sum(1 for t in tokens if t in _SPECIFIC_WORDS or t.isdigit())
    specificity = min(1.0, spec_count / max(1, len(tokens)) * 10)

    # Engagement: length + direct address + question marks.
    address = sum(1 for t in tokens if t in {"you", "your"})
    questions = text.count("?")
    engagement = min(
        1.0,
        (len(tokens) / 100.0) * 0.5 + address * 0.05 + questions * 0.1,
    )

    # Spatial awareness: density of spatial terms.
    spatial_count = sum(1 for t in tokens if t in _SPATIAL_WORDS)
    spatial = min(1.0, spatial_count / max(1, len(tokens)) * 15)

    return Evaluation(
        novelty=round(novelty, 4),
        specificity=round(specificity, 4),
        engagement=round(engagement, 4),
        spatial=round(spatial, 4),
    )
