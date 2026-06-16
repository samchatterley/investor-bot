"""Forward-outcome backfill: turn logged decision-time observations into scored rows.

Each observation in ``logs/experiment_observations.jsonl`` is captured at decision time with no
outcome (``forward_r = None``). Once a forward horizon has *closed* — i.e. the exit bar exists in the
price history available as of the run date — this attaches the path-independent, ATR-normalised
forward return (``experiment.dataset.forward_r``) for that horizon.

Strictly point-in-time: a horizon is filled only when its exit bar is present in the series the caller
supplies (which must stop at the as-of date), so nothing is scored before it is known and no future
bar is ever read. Pure core; the price fetch and file I/O live in ``scripts/backfill_outcomes.py``.

Cost is kept separate from the gross return: each row carries a ``cost_r_estimate`` (round-trip cost
in R units, from the decision-time price and ATR) so the analysis can net it without a cost
assumption being frozen into the stored outcomes.
"""

from __future__ import annotations

from experiment.dataset import forward_r

DEFAULT_HORIZONS: tuple[int, ...] = (1, 3, 5, 10)
DEFAULT_ROUND_TRIP_BPS = 10.0  # ~5 bps each way: spread + slippage on a liquid large/mid-cap name


def _entry_index(dates: list[str], decision_date: str) -> int | None:
    """Index of the decision bar in the (ascending) date series: the bar on, or first after, the
    decision date. None if the decision date is past the end of the series."""
    for i, d in enumerate(dates):
        if d >= decision_date:
            return i
    return None


def cost_r_estimate(
    price: float | None, atr: float | None, round_trip_bps: float = DEFAULT_ROUND_TRIP_BPS
) -> float | None:
    """Round-trip transaction cost expressed in R (ATR) units, or None if price/ATR are unusable."""
    if not price or not atr or atr <= 0:
        return None
    return (price * round_trip_bps / 10_000.0) / atr


def score_observation(
    obs: dict,
    dates: list[str],
    closes: list[float],
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    round_trip_bps: float = DEFAULT_ROUND_TRIP_BPS,
) -> dict:
    """Return a copy of ``obs`` with an ``outcomes`` block: gross forward_r per horizon (None until
    that horizon closes), the cost estimate, and how many horizons are now scored."""
    feats = obs.get("features") or {}
    atr = feats.get("atr")
    price = feats.get("current_price")
    idx = _entry_index(dates, str(obs.get("date", "")))

    gross: dict[str, float | None] = {}
    for h in horizons:
        gross[f"forward_r_{h}d"] = (
            forward_r(closes, idx, h, float(atr)) if (idx is not None and atr) else None
        )

    scored = dict(obs)
    scored["outcomes"] = {
        **gross,
        "cost_r_estimate": cost_r_estimate(price, atr, round_trip_bps),
        "scored_horizons": sum(1 for v in gross.values() if v is not None),
    }
    return scored


def backfill(
    observations: list[dict],
    price_series: dict[str, tuple[list[str], list[float]]],
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    round_trip_bps: float = DEFAULT_ROUND_TRIP_BPS,
) -> list[dict]:
    """Score every observation for which a price series is available.

    ``price_series`` maps symbol -> (ascending date strings, matching closes). Observations whose
    symbol has no series are skipped (left for a later run once prices are available).
    """
    out: list[dict] = []
    for obs in observations:
        series = price_series.get(obs.get("symbol", ""))
        if not series:
            continue
        dates, closes = series
        out.append(
            score_observation(obs, dates, closes, horizons=horizons, round_trip_bps=round_trip_bps)
        )
    return out
