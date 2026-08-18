"""cns_spool trace source — Wesley's bus replies as lessons (plan §3.6).

The distiller gets a real narrow task with a long time-span: Wesley's
recurring reply-shapes on the CNS bus become teaching material, and
recurring shapes compile to ``.nail`` reflexes. The black box becomes a
body, one narrow task at a time.

A "trace" is a USCP packet (as written to ``~/.hermes/cns_inbox`` /
``cns_outbox``) reduced to the distiller's shape: topic + situation +
response text. Recurrence is keyed on the packet's intent + subject, so
the 51st packet of the same shape answers from the reflex cache without
bothering the teacher.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .distiller import TeacherFn, StudentFn, ReflexCache, stage_distill
from .reflex_cache import hash_embedding  # deterministic, no deps

__all__ = ["packet_to_trace", "load_spool", "recurring_shapes", "distill_traces"]

# --------------------------------------------------------------------- #
# USCP packet → distiller trace                                          #
# --------------------------------------------------------------------- #
def packet_to_trace(packet: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Reduce one USCP-v1 packet to a trace (or None if not Wesley-shaped)."""
    header = packet.get("header", {}) if isinstance(packet, dict) else {}
    body = packet.get("body", {}) if isinstance(packet, dict) else {}
    origin = str(header.get("origin_id", ""))
    if "wesley" not in origin.lower():
        return None  # only the ensign's replies are teaching material
    intent = str(header.get("intent", "UNKNOWN"))
    subject = str(body.get("subject", "bus"))
    content = str(body.get("content", "")).strip()
    if not content:
        return None
    return {
        "topic": intent,
        "situation": subject,
        "response": content,
        "signature": f"topic={intent} situation={subject}",
    }


def load_spool(paths) -> List[Dict[str, str]]:
    """Load every USCP json in the given inbox/outbox dirs → traces."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    traces: List[Dict[str, str]] = []
    for p in paths:
        p = Path(p)
        files = sorted(p.glob("*.json")) if p.is_dir() else ([p] if p.is_file() else [])
        for f in files:
            try:
                packet = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            t = packet_to_trace(packet)
            if t:
                traces.append(t)
    return traces


# --------------------------------------------------------------------- #
# Recurrence → reflex                                                    #
# --------------------------------------------------------------------- #
def recurring_shapes(traces: List[Dict[str, str]],
                     minimum: int = 3) -> Dict[str, int]:
    """Signatures that recur at least `minimum` times."""
    counts: Dict[str, int] = {}
    for t in traces:
        counts[t["signature"]] = counts.get(t["signature"], 0) + 1
    return {sig: n for sig, n in counts.items() if n >= minimum}


def distill_traces(traces: List[Dict[str, str]],
                   cache: ReflexCache,
                   minimum: int = 3) -> List[str]:
    """Compile recurring reply-shapes into .nail reflexes.

    Confidence scales with recurrence (3 hits → 0.55, saturating 0.95):
    a shape the ensign repeats is a shape he means. Returns reflex ids.
    """
    counts = recurring_shapes(traces, minimum=minimum)
    reflex_ids: List[str] = []
    by_sig: Dict[str, str] = {}
    for t in traces:
        by_sig.setdefault(t["signature"], t["response"])
    for sig, n in counts.items():
        topic, situation = sig.split(" situation=", 1)
        topic = topic[len("topic="):]
        confidence = min(0.95, 0.4 + n * 0.05)
        rid = cache.store(
            signature=sig,
            response=by_sig[sig],
            confidence=confidence,
            source="cns:spool",
            situation=situation,
            metadata={"topic": topic, "recurrence": n,
                      "trace_source": "cns_spool"},
        )
        reflex_ids.append(rid)
    return reflex_ids
