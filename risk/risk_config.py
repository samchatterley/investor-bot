"""Immutable bundle of trade-exit risk parameters.

Both the backtest engine and live system read these values.  The canonical
live defaults are in config.py; RiskConfig lets the backtest engine or tests
run with alternative parameters without touching global config.

Usage
-----
    # Live defaults
    rc = RiskConfig.from_config()

    # Custom (e.g. walk-forward search)
    rc = RiskConfig(stop_loss_pct=0.05, take_profit_pct=0.20,
                    trailing_stop_pct=5.0, partial_profit_pct=10.0,
                    max_hold_days=4)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfig:
    stop_loss_pct: float  # hard stop: exit when unrealised loss >= this fraction
    take_profit_pct: float  # profit target: exit when unrealised gain >= this fraction
    trailing_stop_pct: float  # trailing stop %: live only; backtest uses fixed stop_loss
    partial_profit_pct: float  # take half position at this % unrealised gain
    max_hold_days: int  # force-exit after this many trading days

    @classmethod
    def from_config(cls) -> RiskConfig:
        """Build a RiskConfig from the current live config values."""
        from config import (
            MAX_HOLD_DAYS,
            PARTIAL_PROFIT_PCT,
            STOP_LOSS_PCT,
            TAKE_PROFIT_PCT,
            TRAILING_STOP_PCT,
        )

        return cls(
            stop_loss_pct=STOP_LOSS_PCT,
            take_profit_pct=TAKE_PROFIT_PCT,
            trailing_stop_pct=TRAILING_STOP_PCT,
            partial_profit_pct=PARTIAL_PROFIT_PCT,
            max_hold_days=MAX_HOLD_DAYS,
        )
