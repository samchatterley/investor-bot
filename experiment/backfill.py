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

import json

from experiment.dataset import forward_r

DEFAULT_HORIZONS: tuple[int, ...] = (1, 3, 5, 10)
DEFAULT_ROUND_TRIP_BPS = 10.0  # ~5 bps each way: spread + slippage on a liquid large/mid-cap name
_ATR_PERIOD = 14  # standard ATR lookback


def _atr_at(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    idx: int | None,
    period: int = _ATR_PERIOD,
) -> float | None:
    """Average true range (PRICE units) over `period` bars ending at `idx`, using only bars ≤ idx.

    Reconstructed here rather than read from the observation: the live snapshot never carries an
    `atr` field, so the decision-time normaliser is computed point-in-time from the price history
    (no bar after the decision bar is touched). None if there isn't enough history.
    """
    if idx is None or idx < period:
        return None
    trs = []
    for i in range(idx - period + 1, idx + 1):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    atr = sum(trs) / len(trs)
    return atr if atr > 0 else None


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
    highs: list[float],
    lows: list[float],
    closes: list[float],
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    round_trip_bps: float = DEFAULT_ROUND_TRIP_BPS,
) -> dict:
    """Return a copy of ``obs`` with an ``outcomes`` block: gross forward_r per horizon (None until
    that horizon closes), the cost estimate, and how many horizons are now scored.

    The ATR normaliser is computed point-in-time from the OHLC history at the decision bar (the live
    snapshot doesn't log atr), so outcomes score from price history alone."""
    feats = obs.get("features") or {}
    price = feats.get("current_price")
    idx = _entry_index(dates, str(obs.get("date", "")))
    atr = _atr_at(highs, lows, closes, idx)

    gross: dict[str, float | None] = {}
    for h in horizons:
        gross[f"forward_r_{h}d"] = (
            forward_r(closes, idx, h, atr) if (idx is not None and atr) else None
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
    price_series: dict[str, tuple[list[str], list[float], list[float], list[float]]],
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    round_trip_bps: float = DEFAULT_ROUND_TRIP_BPS,
) -> list[dict]:
    """Score every observation for which a price series is available.

    ``price_series`` maps symbol -> (ascending dates, highs, lows, closes). Observations whose symbol
    has no series are skipped (left for a later run once prices are available).
    """
    out: list[dict] = []
    for obs in observations:
        series = price_series.get(obs.get("symbol", ""))
        if not series:
            continue
        dates, highs, lows, closes = series
        out.append(
            score_observation(
                obs, dates, highs, lows, closes, horizons=horizons, round_trip_bps=round_trip_bps
            )
        )
    return out


def _scored_horizons(row: dict) -> int:
    return int((row.get("outcomes") or {}).get("scored_horizons", 0) or 0)


def _observation_key(row: dict) -> str:
    """Stable identity of the underlying observation: its full content minus the ``outcomes`` block.

    Observations are append-only and ``score_observation`` copies every field verbatim (only
    ``outcomes`` is added), so the non-outcome content is byte-identical across runs for the same
    observation — and genuinely distinct rows (same symbol+date but different mode/decision_type)
    still differ. Keying on this avoids both collapsing distinct same-day observations and failing to
    match a re-score against its prior row.
    """
    return json.dumps(
        {k: v for k, v in row.items() if k != "outcomes"}, sort_keys=True, default=str
    )


def merge_scored(existing: list[dict], new: list[dict]) -> list[dict]:
    """Merge two scored datasets, keeping the MORE-populated outcome per observation.

    Monotonic by construction: a run whose price fetch failed (all horizons None) can never downgrade
    an observation that was already scored — it simply loses the max() and the prior value is kept.
    This is the failure-safety the runner lacked: it rewrote the whole file each run, so a single
    transient fetch failure wiped every accumulated outcome to None. On ties (same horizon count) the
    newer row wins, so genuine re-scores (e.g. a price revision) still take effect.
    """
    best: dict[str, dict] = {}
    for row in [*existing, *new]:
        key = _observation_key(row)
        if key not in best or _scored_horizons(row) >= _scored_horizons(best[key]):
            best[key] = row
    return list(best.values())
