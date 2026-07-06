"""Live data-feed status recorder — makes silent input degradation visible and structured.

Each critical decision-path fetch records its outcome here as it happens, so:
  1. failures are never silent (record() logs a WARNING on failure), and
  2. the startup health report can surface the degraded feeds as a structured list rather than
     leaving the operator to scrape scattered WARNING lines.

In-memory: the scheduler is one long-running process, so state persists across cycles and resets on
restart (a degraded feed is re-detected on its next fetch). The health check runs at cycle start, so
it reflects the most recent recorded outcome (typically the prior cycle's fetch) — the real-time
signal is the WARNING log; this is the aggregated view.

Distinct from experiment/feed_health, which PROBES feeds on demand (a pre-data gate / manual monitor
that re-fetches). This RECORDS the outcomes of fetches the cycle already makes — zero extra network.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

_STATE: dict[str, dict[str, object]] = {}


def record(feed: str, ok: bool, detail: str = "") -> None:
    """Record a feed fetch outcome. Logs a WARNING on failure so degradation is never silent."""
    _STATE[feed] = {"ok": ok, "ts": datetime.now(UTC).isoformat(), "detail": detail}
    if not ok:
        logger.warning("feed degraded: %s%s", feed, f" — {detail}" if detail else "")


def degraded() -> list[str]:
    """Return the names of feeds whose most recent recorded fetch failed (sorted)."""
    return sorted(f for f, s in _STATE.items() if not s["ok"])


def snapshot() -> dict[str, dict[str, object]]:
    """Return a copy of the full recorded state (for the health report / diagnostics)."""
    return {f: dict(s) for f, s in _STATE.items()}


def reset() -> None:
    """Clear all recorded state (test hook / restart)."""
    _STATE.clear()
