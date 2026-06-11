"""Regime-driven risk policy lookup.

Maps each MarketRegime to a RegimeRiskPolicy that encodes the mechanical
constraints applied at execution time (max orders, confidence bump, size
multiplier).  Centralising these rules here means main.py never needs
regime-name string comparisons.
"""

from dataclasses import dataclass

import config as cfg
from data.market_regime import MarketRegime


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
    MarketRegime.DEFENSIVE_DOWNTREND: RegimeRiskPolicy(
        block_new_buys=False,
        max_orders_per_run=2,
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
    MarketRegime.UNKNOWN: RegimeRiskPolicy(
        block_new_buys=True,
        max_orders_per_run=0,
        min_confidence_bump=99,
        position_size_multiplier=0.0,
    ),
}


def get_regime_policy(regime_name: str) -> RegimeRiskPolicy:
    """Look up the risk policy for a regime by name string.

    Accepts both new names (STRESS_RISK_OFF) and old names (BEAR_DAY, CHOPPY)
    for backward compatibility during transition.
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
    return REGIME_POLICY[regime]
