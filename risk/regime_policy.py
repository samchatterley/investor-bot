"""Regime-driven risk policy lookup.

Maps each MarketRegime to a RegimeRiskPolicy that encodes the mechanical
constraints applied at execution time (max orders, confidence bump, size
multiplier).  Centralising these rules here means main.py never needs
regime-name string comparisons.
"""

import logging
from dataclasses import dataclass

import config as cfg
from data.market_regime import MarketRegime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegimeRiskPolicy:
    block_new_buys: bool
    max_orders_per_run: int
    min_confidence_bump: int
    position_size_multiplier: float


REGIME_POLICY: dict[MarketRegime, RegimeRiskPolicy] = {
    MarketRegime.STRESS_RISK_OFF: RegimeRiskPolicy(
        block_new_buys=True,
        max_orders_per_run=0,
        min_confidence_bump=99,
        position_size_multiplier=0.0,
    ),
    MarketRegime.HIGH_VOL_DOWNTREND: RegimeRiskPolicy(
        block_new_buys=False,
        max_orders_per_run=1,
        min_confidence_bump=2,
        position_size_multiplier=0.40,
    ),
    # ADR-006: entry must agree with the regime-change exit, which force-closes any long
    # held <2 days in this regime. Opening longs here only churns them out next run (the
    # 06-18..06-24 bleed: 27% WR, -0.89%/trade). Block new entries — the bot holds cash
    # until the short side (ADR-006 part B) provides a downside tool. max_orders_per_run=0
    # is a second, independent guard behind block_new_buys.
    MarketRegime.DEFENSIVE_DOWNTREND: RegimeRiskPolicy(
        block_new_buys=True,
        max_orders_per_run=0,
        min_confidence_bump=0,
        position_size_multiplier=0.75,
    ),
    MarketRegime.BULL_TREND: RegimeRiskPolicy(
        block_new_buys=False,
        max_orders_per_run=cfg.MAX_ORDERS_PER_RUN,
        min_confidence_bump=0,
        position_size_multiplier=1.0,
    ),
    MarketRegime.NEUTRAL_CHOP: RegimeRiskPolicy(
        block_new_buys=False,
        max_orders_per_run=2,
        min_confidence_bump=0,
        position_size_multiplier=0.75,
    ),
    # CREDIT_STRESS: credit-spread tightening precedes equity stress; treat same
    # risk budget as HIGH_VOL — one order max, elevated confidence bar, 40% size.
    MarketRegime.CREDIT_STRESS: RegimeRiskPolicy(
        block_new_buys=False,
        max_orders_per_run=1,
        min_confidence_bump=2,
        position_size_multiplier=0.40,
    ),
    # LATE_CYCLE_BULL: bull price action but macro warns (inverted curve / breadth
    # divergence).  Same signal blocks as NEUTRAL_CHOP; slight confidence uplift.
    MarketRegime.LATE_CYCLE_BULL: RegimeRiskPolicy(
        block_new_buys=False,
        max_orders_per_run=2,
        min_confidence_bump=1,
        position_size_multiplier=0.75,
    ),
    # RECOVERY: positive 5d momentum while still in ≥5% drawdown.  More
    # permissive than DEFENSIVE_DOWNTREND but not full bull sizing yet.
    MarketRegime.RECOVERY: RegimeRiskPolicy(
        block_new_buys=False,
        max_orders_per_run=2,
        min_confidence_bump=0,
        position_size_multiplier=0.65,
    ),
    MarketRegime.UNKNOWN: RegimeRiskPolicy(
        block_new_buys=True,
        max_orders_per_run=0,
        min_confidence_bump=99,
        position_size_multiplier=0.0,
    ),
}

# Totality guard: every MarketRegime member must have a policy entry.
# This fires at import time so a missing regime is caught immediately.
_missing = set(MarketRegime) - set(REGIME_POLICY)
assert not _missing, f"REGIME_POLICY missing entries for: {_missing}"

_UNKNOWN_POLICY = REGIME_POLICY[MarketRegime.UNKNOWN]


def get_regime_policy(regime_name: str) -> RegimeRiskPolicy:
    """Look up the risk policy for a regime by name string.

    Accepts both new names (STRESS_RISK_OFF) and old names (BEAR_DAY, CHOPPY)
    for backward compatibility during transition.  Unknown regime names fall back
    to UNKNOWN policy (block_new_buys=True) with a logged alert rather than
    raising KeyError.
    """
    _LEGACY_MAP = {
        "BEAR_DAY": MarketRegime.STRESS_RISK_OFF,
        "HIGH_VOL": MarketRegime.HIGH_VOL_DOWNTREND,
        "BULL_TRENDING": MarketRegime.BULL_TREND,
        "CHOPPY": MarketRegime.NEUTRAL_CHOP,
    }
    try:
        regime = MarketRegime(regime_name)
    except ValueError:
        regime = _LEGACY_MAP.get(regime_name, MarketRegime.UNKNOWN)
    policy = REGIME_POLICY.get(regime)
    if policy is None:
        logger.error(
            f"get_regime_policy: no policy for regime '{regime_name}' — "
            "falling back to UNKNOWN (block_new_buys=True). Add it to REGIME_POLICY."
        )
        return _UNKNOWN_POLICY
    return policy
