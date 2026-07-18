"""Lookahead guard: an automated audit that a point-in-time computation never reads the future.

Substrate brick 2. The codebase is full of "so there is no lookahead" *claims* -- replay.py slices
frames to <= T, historical_fundamentals walks sorted events, backfill fills only closed horizons -- but
every one is a comment, not a test. This module makes the claim falsifiable.

The technique is **future-data poisoning**: run a deterministic computation for date T twice -- once on
the real data, once on data with everything strictly after T removed -- and compare. A correct
point-in-time computation ignores the future, so the two outputs are identical; any divergence (or a
crash only on the poisoned run) is a leak, localized to T. This is the falsification test the validation
substrate demands of itself, so the guard even ships with a **canary**: a deliberately-leaky computation
it must catch, proving the detector detects.

Pure and offline: the computation under audit is injected, so the guard has no data or network deps.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

# A computation under audit: (data, as_of_date) -> an equality-comparable result (what was "known" at T).
ComputeFn = Callable[[Any, str], Any]
# A poison: (data, as_of_date) -> data with everything strictly after as_of removed/masked.
PoisonFn = Callable[[Any, str], Any]


@dataclass
class Leak:
    date: str
    detail: str


def drop_after(records: list[dict], as_of: str, *, date_key: str = "date") -> list[dict]:
    """Poison for record lists: keep only rows dated on/before ``as_of`` (the future did not exist).

    A point-in-time-honest computation slices to <= as_of anyway, so dropping the future is a no-op for
    it; a computation that reaches past as_of will diverge or crash."""
    return [r for r in records if r.get(date_key, "") <= as_of]


def audit_no_lookahead(
    compute_fn: ComputeFn,
    data: Any,
    dates: Sequence[str],
    *,
    poison: PoisonFn = drop_after,
) -> list[Leak]:
    """Return the dates at which ``compute_fn`` read the future (empty list == provably clean here).

    For each T, compare ``compute_fn(data, T)`` with ``compute_fn(poison(data, T), T)``. Identical means
    the future was never used; a divergence -- or an exception raised only on the poisoned input -- is a
    leak. ``compute_fn``'s output must be equality-comparable (a scalar, tuple, dict, ...)."""
    leaks: list[Leak] = []
    for t in dates:
        clean = compute_fn(data, t)
        try:
            poisoned = compute_fn(poison(data, t), t)
        except Exception as exc:  # noqa: BLE001 - a crash only when the future is removed IS the leak
            leaks.append(
                Leak(date=t, detail=f"computation needed future data (raised {type(exc).__name__})")
            )
            continue
        if clean != poisoned:
            leaks.append(
                Leak(
                    date=t,
                    detail=f"output changed when future was masked: {clean!r} != {poisoned!r}",
                )
            )
    return leaks


def leaky_canary(
    records: list[dict], as_of: str, *, date_key: str = "date", value_key: str = "value"
) -> float:
    """A deliberately-leaky computation (the guard's self-test): it reads the FIRST value strictly after
    ``as_of`` -- pure lookahead. ``audit_no_lookahead`` MUST flag this; if it ever stops, the detector is
    broken. Returns 0.0 when there is no future (the last date can't leak)."""
    future = sorted((r for r in records if r[date_key] > as_of), key=lambda r: r[date_key])
    return float(future[0][value_key]) if future else 0.0
