from exocortex import ReflexCache
from exocortex.distiller import (
    run_iteration,
    stage_distill,
    stage_evaluate,
    stage_student,
    stage_teacher,
    stage_update_prompt,
)


def stub_teacher(prompt: str) -> str:
    return f"Lesson for: {prompt}"


def stub_student(prompt: str) -> str:
    # When primed with a lesson, emit a longer, more specific response.
    if "[LESSON]" in prompt:
        return "taught response with north, east, and precise measurements"
    return "baseline response"


def test_stage_teacher_and_student():
    lesson = stage_teacher("prompt", stub_teacher)
    assert "Lesson for" in lesson

    baseline = stage_student("prompt", stub_student)
    taught = stage_student("prompt", stub_student, lesson=lesson)
    assert baseline != taught


def test_stage_evaluate_positive_delta():
    baseline = "it is big"
    taught = "it is 5 metres wide and 2 metres tall, located north of the door"
    base, taught_scores, delta = stage_evaluate(baseline, taught)
    assert delta > 0
    assert taught_scores.spatial > base.spatial


def test_stage_update_prompt_requires_consecutive_positives():
    assert stage_update_prompt("roblox", 2, threshold=3) is None
    assert stage_update_prompt("roblox", 3, threshold=3) is not None


def test_stage_distill_only_compiles_positive_delta():
    cache = ReflexCache()
    rid = stage_distill("roblox", "strict mode", "use --!strict", cache, delta=0.2)
    assert rid is not None
    assert cache.size() == 1

    rid2 = stage_distill("roblox", "strict mode", "bad lesson", cache, delta=-0.1)
    assert rid2 is None


def test_run_iteration_full_loop():
    cache = ReflexCache()
    result = run_iteration(
        topic="roblox",
        situation="luau strict mode",
        prompt="how do I enable strict mode?",
        teacher_fn=stub_teacher,
        student_fn=stub_student,
        cache=cache,
    )
    assert result["delta"] > 0
    assert result["reflex_id"] is not None
    assert cache.size() == 1


def test_run_iteration_resets_consecutive_on_negative_delta():
    cache = ReflexCache()

    def bad_student(prompt: str) -> str:
        return "short"

    result = run_iteration(
        topic="roblox",
        situation="luau",
        prompt="how?",
        teacher_fn=stub_teacher,
        student_fn=bad_student,
        cache=cache,
        consecutive_positives=2,
    )
    assert result["delta"] <= 0
    assert result["consecutive_positives"] == 0
    assert result["prompt_update"] is None
