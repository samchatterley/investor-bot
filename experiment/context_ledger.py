"""As-of context ledger: lookahead-safe admissibility for context items (EXPERIMENT.md section 13).

The dominant false-positive risk for the contextual arm is leakage: letting a context item into Arm 3
that we could not actually have seen before the decision. This module enforces the admissibility rule:
a context item is admissible only if it was provably observed before the decision, with a safety
buffer, and with unambiguous timestamps. Anything missing, late, or internally inconsistent is
quarantined and excluded from the primary metric.

Pure logic, no I/O. All timestamps must be timezone-consistent (use UTC throughout).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

# timestamp_confidence labels (EXPERIMENT.md section 13).
TIMESTAMP_CLEAN = "clean"
TIMESTAMP_UNCERTAIN = "uncertain"
TIMESTAMP_REJECTED = "rejected"

_DEFAULT_SAFETY_BUFFER_SECONDS = 300  # 5-minute buffer to avoid borderline-timestamp leakage


@dataclass
class ContextItem:
    """A single timestamped context observation (provenance fields per EXPERIMENT.md section 13)."""

    symbol: str
    decision_id: str
    decision_time: datetime
    source: str
    source_id: str
    text: str
    provider_published_at: datetime | None = None
    provider_seen_at: datetime | None = None
    retrieved_at: datetime | None = None


@dataclass
class Admissibility:
    admissible: bool
    timestamp_confidence: str
    reason: str


def assess_admissibility(
    item: ContextItem,
    safety_buffer_seconds: int = _DEFAULT_SAFETY_BUFFER_SECONDS,
) -> Admissibility:
    """Decide whether a context item may enter Arm 3, with a timestamp_confidence label.

    Rejected (hard): missing required timestamps; retrieved after the decision; seen after the
    decision. Uncertain (excluded from the primary): seen within the safety buffer of the decision;
    published after it was seen (a possible backfill or clock-skew inconsistency). Otherwise clean.
    """
    dt = item.decision_time
    if item.provider_seen_at is None or item.retrieved_at is None:
        return Admissibility(False, TIMESTAMP_REJECTED, "missing provider_seen_at or retrieved_at")
    if item.retrieved_at > dt:
        return Admissibility(False, TIMESTAMP_REJECTED, "retrieved after decision_time (leakage)")
    if item.provider_seen_at > dt:
        return Admissibility(False, TIMESTAMP_REJECTED, "seen after decision_time (leakage)")
    cutoff = dt - timedelta(seconds=safety_buffer_seconds)
    if item.provider_seen_at > cutoff:
        return Admissibility(
            False, TIMESTAMP_UNCERTAIN, "seen within the safety buffer of decision_time"
        )
    if (
        item.provider_published_at is not None
        and item.provider_published_at > item.provider_seen_at
    ):
        return Admissibility(
            False,
            TIMESTAMP_UNCERTAIN,
            "published_at after seen_at (possible backfill / clock skew)",
        )
    return Admissibility(True, TIMESTAMP_CLEAN, "ok")


def filter_admissible(
    items: list[ContextItem],
    safety_buffer_seconds: int = _DEFAULT_SAFETY_BUFFER_SECONDS,
) -> list[ContextItem]:
    """Return only the items admissible for the primary metric (clean timestamps)."""
    return [it for it in items if assess_admissibility(it, safety_buffer_seconds).admissible]
