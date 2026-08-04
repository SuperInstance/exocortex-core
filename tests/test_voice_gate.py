from exocortex.voice_gate import RouteTarget, Trigger, VoiceGate


def test_exact_match_routes_to_reflex():
    gate = VoiceGate()
    gate.register(
        Trigger(category="light", phrase="turn on the light", action="reflex:light_on", match_type="exact")
    )
    decision = gate.classify("turn on the light")
    assert decision.target == RouteTarget.REFLEX
    assert decision.confidence == 1.0
    assert decision.trigger is not None
    assert decision.urgent is False


def test_substring_match_routes_to_local():
    gate = VoiceGate()
    gate.register(
        Trigger(category="weather", phrase="weather", action="local", match_type="substring")
    )
    decision = gate.classify("what is the weather today")
    assert decision.target == RouteTarget.LOCAL
    assert decision.confidence == 0.85
    assert decision.matched_phrase == "weather"


def test_unknown_transcript_routes_to_cloud():
    gate = VoiceGate()
    gate.register(Trigger(category="weather", phrase="weather", action="local"))
    decision = gate.classify("quantum entanglement in luau")
    assert decision.target == RouteTarget.CLOUD
    assert decision.confidence == 0.2
    assert decision.trigger is None


def test_urgency_signal_detected():
    gate = VoiceGate()
    decision = gate.classify("stop everything right now")
    assert decision.urgent is True
    assert decision.target == RouteTarget.CLOUD


def test_urgency_overrides_but_keeps_exact_match():
    gate = VoiceGate()
    gate.register(
        Trigger(category="safety", phrase="stop", action="reflex:emergency_stop", match_type="exact")
    )
    decision = gate.classify("stop")
    assert decision.urgent is True
    assert decision.target == RouteTarget.REFLEX
