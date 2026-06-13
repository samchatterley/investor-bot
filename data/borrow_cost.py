"""Stock-borrow cost estimation for short positions.

No paid cost-to-borrow feed is available, so we estimate the annualized borrow
rate from short-interest signals — the strongest *free* proxy. Heavily-shorted,
crowded names are expensive- or hard-to-borrow (HTB); liquid, lightly-shorted
names borrow at the general-collateral (GC) rate.

This module exists because every short backtest before v1.99 modelled borrow as
free, systematically overstating short P&L. The tiers below are calibrated to
typical US equity borrow markets (annualized, as a decimal):

    short_pct_float < 5%    → GC          ~0.5%/yr
    5–15%                   → moderate     ~3%/yr
    15–30%                  → elevated    ~10%/yr
    30–50%                  → hard         ~30%/yr
    > 50%                   → special     ~80%/yr (often unborrowable)

`short_pct_float` is the primary input. When it is unavailable, `short_ratio`
(days-to-cover) is used as a coarse fallback. When neither is present we assume
the GC rate — the name is not flagged as crowded, so cheap borrow is the
reasonable prior.

Public API:
    estimate_borrow_rate(short_pct_float, short_ratio=None) -> float   # annualized decimal
    is_hard_to_borrow(short_pct_float, short_ratio=None) -> bool
    borrow_cost_usd(rate_annual, notional, days_held) -> float
    borrow_rate_for_symbol(symbol, short_interest) -> float            # dict-of-dicts convenience
"""

from __future__ import annotations

# ── Calibration constants ─────────────────────────────────────────────────────
# General-collateral rate: the floor cost applied to any easy-to-borrow name.
GC_RATE: float = 0.005  # 0.5%/yr

# (upper_bound_pct_of_float, annualized_rate) tiers, evaluated in ascending order.
# A name with short_pct_float at or below the bound takes that tier's rate.
_PCT_FLOAT_TIERS: tuple[tuple[float, float], ...] = (
    (0.05, GC_RATE),  # < 5%   → GC
    (0.15, 0.03),  # 5–15%  → 3%
    (0.30, 0.10),  # 15–30% → 10%
    (0.50, 0.30),  # 30–50% → 30% (hard to borrow)
)
_SPECIAL_RATE: float = 0.80  # > 50% of float → special / often unborrowable

# short_ratio (days-to-cover) fallback tiers when short_pct_float is absent.
_SHORT_RATIO_TIERS: tuple[tuple[float, float], ...] = (
    (3.0, GC_RATE),  # < 3 days  → GC
    (7.0, 0.03),  # 3–7 days  → 3%
    (10.0, 0.10),  # 7–10 days → 10%
)
_SHORT_RATIO_SPECIAL_RATE: float = 0.30  # > 10 days-to-cover → hard

# A position is hard-to-borrow (and should be avoided by the short path) when its
# estimated borrow rate is at or above this threshold — borrow can spike, the
# lender can recall, and crowded shorts are squeeze fuel.
HTB_RATE_THRESHOLD: float = 0.30

_DAYS_PER_YEAR: int = 252  # trading days — matches the backtest holding-period convention


def estimate_borrow_rate(
    short_pct_float: float | None,
    short_ratio: float | None = None,
) -> float:
    """Return the estimated annualized borrow rate (decimal) for a name.

    Uses short_pct_float tiers when available; falls back to short_ratio
    (days-to-cover) tiers; defaults to the GC rate when neither is present.
    """
    if short_pct_float is not None:
        pct = float(short_pct_float)
        # yfinance reports short_pct_float as a fraction (0.18 = 18%); guard against
        # an accidental percentage form (18.0) by normalising values > 1.5.
        if pct > 1.5:
            pct = pct / 100.0
        for upper, rate in _PCT_FLOAT_TIERS:
            if pct <= upper:
                return rate
        return _SPECIAL_RATE

    if short_ratio is not None:
        ratio = float(short_ratio)
        for upper, rate in _SHORT_RATIO_TIERS:
            if ratio <= upper:
                return rate
        return _SHORT_RATIO_SPECIAL_RATE

    return GC_RATE


def is_hard_to_borrow(
    short_pct_float: float | None,
    short_ratio: float | None = None,
) -> bool:
    """True when the estimated borrow rate is at or above the HTB threshold."""
    return estimate_borrow_rate(short_pct_float, short_ratio) >= HTB_RATE_THRESHOLD


def borrow_cost_usd(rate_annual: float, notional: float, days_held: float) -> float:
    """Dollar borrow cost for holding ``notional`` short at ``rate_annual`` for ``days_held``.

    Pro-rated over a 252-trading-day year. Never negative.
    """
    if rate_annual <= 0 or notional <= 0 or days_held <= 0:
        return 0.0
    return notional * rate_annual * (days_held / _DAYS_PER_YEAR)


def borrow_rate_for_symbol(symbol: str, short_interest: dict[str, dict | None]) -> float:
    """Convenience: look up a symbol's short-interest record and estimate its borrow rate.

    ``short_interest`` is the dict-of-dicts produced by data.short_interest
    (``{symbol: {"short_pct_float": float|None, "short_ratio": float|None}}``).
    Missing symbols fall back to the GC rate.
    """
    record = short_interest.get(symbol)
    if not record:
        return GC_RATE
    return estimate_borrow_rate(record.get("short_pct_float"), record.get("short_ratio"))
