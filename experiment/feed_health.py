"""Data-feed health gate: detect silently-degraded feeds before collecting experiment data.

The bot degrades gracefully on any data failure (a broken feed returns None / empty / a neutral
default), so broken feeds are invisible in normal operation — this is exactly how the AAII (missing
xlrd) and 8-K (wrong document) bugs hid for weeks. This module is the pure classification core;
``scripts/feed_health_check.py`` wires the live feeds and runs it.

Use it as a pre-data gate (do not start collecting experiment observations until the feeds the
experiment depends on are green) and as ongoing monitoring (feeds rot: APIs start 403-ing, layouts
drift, optional deps vanish — both bugs we found had *drifted* from working to broken).

A feed is classified by its probe outcome:
  OK       — returned live, plausible data
  EMPTY    — returned None or an empty collection (often a silent degradation)
  DEGRADED — returned data that fails a plausibility check (e.g. AAII percentages all NaN)
  STALE    — returned data older than its freshness budget
  ERROR    — the fetch raised
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

OK = "ok"
EMPTY = "empty"
DEGRADED = "degraded"
STALE = "stale"
ERROR = "error"

_EMPTY_TYPES = (str, bytes, list, tuple, set, frozenset, dict)


@dataclass
class FeedResult:
    name: str
    status: str
    detail: str = ""

    @property
    def healthy(self) -> bool:
        return self.status == OK


def check_feed(
    name: str,
    fetch: Callable[[], object],
    assess: Callable[[object], tuple[str, str]] | None = None,
) -> FeedResult:
    """Run one feed probe and classify it.

    ``fetch`` returns a sample value; ``assess`` (optional) judges a non-empty value and returns a
    ``(status, detail)`` pair. check_feed itself handles the universal degradation signatures
    (exception, None, empty collection) so each feed's ``assess`` only encodes its own plausibility
    and freshness rules.
    """
    try:
        value = fetch()
    except Exception as exc:
        return FeedResult(name, ERROR, f"{type(exc).__name__}: {exc}")
    if value is None:
        return FeedResult(name, EMPTY, "returned None")
    if isinstance(value, _EMPTY_TYPES) and len(value) == 0:
        return FeedResult(name, EMPTY, "empty result")
    if assess is None:
        return FeedResult(name, OK, "")
    status, detail = assess(value)
    return FeedResult(name, status, detail)


Probe = tuple[str, Callable[[], object], Callable[[object], tuple[str, str]] | None]


def run_health_checks(probes: list[Probe]) -> list[FeedResult]:
    """Run every probe (each isolated — one feed's failure never aborts the rest)."""
    return [check_feed(name, fetch, assess) for name, fetch, assess in probes]


def summarise(results: list[FeedResult]) -> tuple[int, int, bool]:
    """Return (healthy_count, needs_attention_count, all_green)."""
    ok = sum(1 for r in results if r.healthy)
    bad = len(results) - ok
    return ok, bad, bad == 0


def format_report(results: list[FeedResult]) -> str:
    """Human-readable per-feed report (unhealthy feeds listed first)."""
    label = {OK: "OK   ", EMPTY: "EMPTY", DEGRADED: "DEGRD", STALE: "STALE", ERROR: "ERROR"}
    lines = ["Data-feed health check", "=" * 64]
    for r in sorted(results, key=lambda x: (x.healthy, x.name)):
        lines.append(f"  [{label.get(r.status, r.status)}] {r.name:26} {r.detail}")
    ok, bad, all_green = summarise(results)
    lines.append("=" * 64)
    verdict = "ALL GREEN" if all_green else f"{bad} feed(s) need attention"
    lines.append(f"  {ok} healthy / {len(results)} total — {verdict}")
    return "\n".join(lines)
