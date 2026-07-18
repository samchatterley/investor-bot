"""Canonical point-in-time accessor: the single tested place the "knowable by date T" rule lives.

Substrate brick 2 (Phase B). The bot already has correct-but-scattered point-in-time logic --
historical_fundamentals walks sorted event lists, universe_history filters by first-tradeable date --
and each re-implements "use only what was known by T" while merely *asserting* correctness. This module
gives them one construction-safe selector to build on (``visible_as_of``) and one tripwire to verify
outputs (``assert_no_future``), so the lookahead guard has a single auditable contract instead of N
hand-rolled ``<= T`` filters that each have to be trusted separately.

ISO date strings (YYYY-MM-DD) sort lexically, so ``str`` and ``datetime.date`` keys are interchangeable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import date, timedelta
from typing import Any


class LookaheadError(AssertionError):
    """A record dated strictly after the as-of date leaked into a point-in-time view."""


def _iso(value: Any) -> str:
    """Normalise a date-like (str | date) to an ISO 'YYYY-MM-DD' string for lexical comparison."""
    return value if isinstance(value, str) else value.isoformat()


def visible_as_of[T](
    records: Iterable[T],
    on_date: Any,
    *,
    date_of: Callable[[T], Any],
    within_days: int | None = None,
) -> list[T]:
    """Records knowable at ``on_date``: ``date_of(r) <= on_date`` (and within ``within_days`` if given).

    The invariant is enforced *by construction* -- a future-dated record is never admitted -- so callers
    get point-in-time correctness instead of hand-rolling (and occasionally fumbling) the ``<= T``
    filter. Input order is preserved."""
    on = _iso(on_date)
    lo: str | None = None
    if within_days is not None:
        lo = (date.fromisoformat(on) - timedelta(days=within_days)).isoformat()
    out: list[T] = []
    for r in records:
        d = _iso(date_of(r))
        if d > on:
            continue
        if lo is not None and d < lo:
            continue
        out.append(r)
    return out


def latest_as_of[T](records: Iterable[T], on_date: Any, *, date_of: Callable[[T], Any]) -> T | None:
    """The single most-recent record knowable at ``on_date`` (``None`` if there is none)."""
    visible = visible_as_of(records, on_date, date_of=date_of)
    return max(visible, key=lambda r: _iso(date_of(r))) if visible else None


def assert_no_future[T](records: Iterable[T], on_date: Any, *, date_of: Callable[[T], Any]) -> None:
    """Tripwire: raise ``LookaheadError`` if any record post-dates ``on_date``.

    Use it to verify a point-in-time provider's *output* -- belt-and-braces over ``visible_as_of``, and
    the hook the lookahead guard drives against the existing providers."""
    on = _iso(on_date)
    offenders = [r for r in records if _iso(date_of(r)) > on]
    if offenders:
        raise LookaheadError(f"{len(offenders)} record(s) post-date {on}")
