"""cns_traces tests — Wesley's bus replies as lessons (plan §3.6)."""
import json
from pathlib import Path

import pytest

from exocortex.cns_traces import (
    packet_to_trace, load_spool, recurring_shapes, distill_traces)
from exocortex.reflex_cache import ReflexCache


def wesley_packet(i: int, intent="QUERY", subject="bus") -> dict:
    return {
        "header": {"type": "USCP-v1", "origin_id": "wesley",
                   "timestamp": f"2026-08-18T09:{i:02d}:00Z",
                   "priority": "MEDIUM", "intent": intent},
        "body": {"subject": subject, "content": f"Wesley reply number {i}: the wiki grew again.",
                 "source": "wesley"},
    }


def other_packet() -> dict:
    p = wesley_packet(0)
    p["header"]["origin_id"] = "hermes-cns"
    return p


def test_only_wesley_packets_become_traces():
    assert packet_to_trace(wesley_packet(1)) is not None
    assert packet_to_trace(other_packet()) is None
    empty = wesley_packet(2); empty["body"]["content"] = " "
    assert packet_to_trace(empty) is None


def test_load_spool(tmp_path: Path):
    for i in range(5):
        (tmp_path / f"w{i}.json").write_text(json.dumps(wesley_packet(i)))
    (tmp_path / "x.json").write_text(json.dumps(other_packet()))
    (tmp_path / "bad.json").write_text("{not json")
    traces = load_spool(tmp_path)
    assert len(traces) == 5
    assert traces[0]["topic"] == "QUERY"


def test_recurring_shapes():
    traces = [packet_to_trace(wesley_packet(i)) for i in range(5)]
    traces += [packet_to_trace(wesley_packet(i, intent="STATUS_REPORT")) for i in range(2)]
    rec = recurring_shapes(traces, minimum=3)
    assert "topic=QUERY situation=bus" in rec and rec["topic=QUERY situation=bus"] == 5
    assert "topic=STATUS_REPORT situation=bus" not in rec


def test_50_traces_compile_a_reflex_that_answers_the_51st(tmp_path: Path):
    """The plan's acceptance: ×50 → ≥1 .nail reflex; the 51st answers from cache."""
    cache = ReflexCache(embed_fn=None)  # default hash embedding
    traces = [packet_to_trace(wesley_packet(i)) for i in range(50)]
    ids = distill_traces(traces, cache, minimum=3)
    assert len(ids) >= 1
    sig = traces[0]["signature"]
    hit = cache.lookup(sig)
    assert hit is not None
    assert hit.source == "cns:spool"
    # the 51st packet of the same shape answers without the teacher
    fifty_first = packet_to_trace(wesley_packet(50))
    assert cache.lookup(fifty_first["signature"]) is not None
    # confidence scales with recurrence (50 hits → saturated near 0.95)
    assert hit.confidence >= 0.9
