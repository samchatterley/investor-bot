"""Composite fear/greed index built from available market data.

Combines 5 components into a 0–100 score:
  - VIX level (fear = high VIX)
  - AAII bears % (fear = high bears)
  - NH/NL ratio (fear = low ratio)
  - SPY 20-day return (fear = falling)
  - Breadth (fear = narrow breadth)

Scores <20 = extreme fear (contrarian long signal).
Scores >80 = excessive greed (caution, dampen longs).

All components degrade gracefully to neutral (50) when data is unavailable.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Component weights (sum to 1.0)
_VIX_WEIGHT = 0.30
_AAII_WEIGHT = 0.25
_NHL_WEIGHT = 0.20
_MOMENTUM_WEIGHT = 0.15
_BREADTH_WEIGHT = 0.10


def _vix_component(vix: float | None) -> float:
    """Convert VIX level to 0–100 component (high VIX = low score = fear)."""
    if vix is None:
        return 50.0
    # VIX 10 → score 90 (greed); VIX 40 → score 10 (fear); linear interpolation
    score = 90.0 - (vix - 10.0) * (80.0 / 30.0)
    return float(max(0.0, min(100.0, score)))


def _aaii_component(bears_pct: float | None) -> float:
    """Convert AAII bears % to 0–100 component (high bears = low score = fear)."""
    if bears_pct is None:
        return 50.0
    # Bears 20% → score 90; Bears 55% → score 10; linear
    score = 90.0 - (bears_pct - 20.0) * (80.0 / 35.0)
    return float(max(0.0, min(100.0, score)))


def _nhl_component(nhl_ratio: float | None) -> float:
    """Convert NH/NL ratio to 0–100 component (low ratio = fear)."""
    if nhl_ratio is None:
        return 50.0
    # Ratio 0.1 → score 10; Ratio 5.0 → score 90
    score = 10.0 + (nhl_ratio - 0.1) * (80.0 / 4.9)
    return float(max(0.0, min(100.0, score)))


def _momentum_component(spy_ret_20d: float | None) -> float:
    """Convert SPY 20d return to 0–100 component (negative = fear)."""
    if spy_ret_20d is None:
        return 50.0
    # -10% return → score 10; +10% return → score 90; linear
    score = 50.0 + spy_ret_20d * (40.0 / 10.0)
    return float(max(0.0, min(100.0, score)))


def _breadth_component(breadth_pct_above_50d: float | None) -> float:
    """Convert breadth % above 50d SMA to 0–100 component."""
    if breadth_pct_above_50d is None:
        return 50.0
    # 20% above → score 20; 80% above → score 80; linear
    score = breadth_pct_above_50d
    return float(max(0.0, min(100.0, score)))


def compute_fear_greed_index(
    vix: float | None = None,
    aaii_bears_pct: float | None = None,
    nhl_ratio: float | None = None,
    spy_ret_20d: float | None = None,
    breadth_pct_above_50d: float | None = None,
) -> float:
    """Compute 0–100 composite fear/greed index.

    0 = extreme fear, 100 = extreme greed, 50 = neutral.
    Missing components default to 50 (neutral contribution).

    Args:
        vix:                  VIX spot level (e.g. 20.0)
        aaii_bears_pct:       AAII bears percentage (e.g. 35.0 means 35%)
        nhl_ratio:            New-highs / (new-lows + 1) ratio
        spy_ret_20d:          SPY 20-day price return in percent (e.g. -3.5)
        breadth_pct_above_50d: % of stocks above 50d SMA (0–100)

    Returns:
        float in [0, 100]
    """
    v_vix = _vix_component(vix)
    v_aaii = _aaii_component(aaii_bears_pct)
    v_nhl = _nhl_component(nhl_ratio)
    v_mom = _momentum_component(spy_ret_20d)
    v_breadth = _breadth_component(breadth_pct_above_50d)

    composite = (
        v_vix * _VIX_WEIGHT
        + v_aaii * _AAII_WEIGHT
        + v_nhl * _NHL_WEIGHT
        + v_mom * _MOMENTUM_WEIGHT
        + v_breadth * _BREADTH_WEIGHT
    )
    return round(float(max(0.0, min(100.0, composite))), 1)


def is_extreme_fear(score: float) -> bool:
    """True when composite score < 20 (extreme fear → contrarian long signal)."""
    return score < 20.0


def is_excessive_greed(score: float) -> bool:
    """True when composite score > 80 (excessive greed → dampen longs)."""
    return score > 80.0
