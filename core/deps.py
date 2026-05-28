"""
Dependency container for the trading pipeline.

TradingDeps holds every external module and callable that pipeline functions
need. Constructing it once per run (via TradingDeps.production()) and passing
it through the pipeline makes dependencies explicit and simple to test:
tests build a TradingDeps with MagicMock fields instead of patching 700
module attributes.

RunConfig is a frozen snapshot of the configuration values the pipeline
consumes, captured at run start via RunConfig.from_config(). Being frozen
(immutable) means pipeline functions cannot accidentally mutate shared state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    """Immutable snapshot of config values consumed by pipeline functions."""

    bear_market_spy_threshold: float
    cash_reserve_pct: float
    earnings_warning_days: int
    is_paper: bool
    lookback_days: int
    max_daily_loss_usd: float
    max_daily_notional_usd: float
    max_deployed_usd: float
    max_experiment_drawdown_usd: float
    max_hold_days: int
    max_orders_per_run: int
    max_positions: int
    max_price_usd: float
    max_sector_positions: int
    max_short_hedge_ratio: float
    max_short_hold_days: int
    max_short_positions: int
    max_single_order_usd: float
    min_confidence: int
    min_price_usd: float
    short_size_scale: float
    signal_max_hold_days: dict  # {signal_name: max_days}
    small_account_mode: bool
    top_movers_count: int
    trailing_stop_pct: float

    @classmethod
    def from_config(cls) -> RunConfig:
        """Capture the current config module state into an immutable snapshot."""
        import config

        return cls(
            bear_market_spy_threshold=config.BEAR_MARKET_SPY_THRESHOLD,
            cash_reserve_pct=config.CASH_RESERVE_PCT,
            earnings_warning_days=config.EARNINGS_WARNING_DAYS,
            is_paper=config.IS_PAPER,
            lookback_days=config.LOOKBACK_DAYS,
            max_daily_loss_usd=config.MAX_DAILY_LOSS_USD,
            max_daily_notional_usd=config.MAX_DAILY_NOTIONAL_USD,
            max_deployed_usd=config.MAX_DEPLOYED_USD,
            max_experiment_drawdown_usd=config.MAX_EXPERIMENT_DRAWDOWN_USD,
            max_hold_days=config.MAX_HOLD_DAYS,
            max_orders_per_run=config.MAX_ORDERS_PER_RUN,
            max_positions=config.MAX_POSITIONS,
            max_price_usd=config.MAX_PRICE_USD,
            max_sector_positions=config.MAX_SECTOR_POSITIONS,
            max_short_hedge_ratio=config.MAX_SHORT_HEDGE_RATIO,
            max_short_hold_days=config.MAX_SHORT_HOLD_DAYS,
            max_short_positions=config.MAX_SHORT_POSITIONS,
            max_single_order_usd=config.MAX_SINGLE_ORDER_USD,
            min_confidence=config.MIN_CONFIDENCE,
            min_price_usd=config.MIN_PRICE_USD,
            short_size_scale=config.SHORT_SIZE_SCALE,
            signal_max_hold_days=dict(config.SIGNAL_MAX_HOLD_DAYS),
            small_account_mode=config.SMALL_ACCOUNT_MODE,
            top_movers_count=config.TOP_MOVERS_COUNT,
            trailing_stop_pct=config.TRAILING_STOP_PCT,
        )


@dataclass
class TradingDeps:
    """All external dependencies for the trading pipeline.

    Pass a single instance through _run_inner and into each pipeline function
    instead of importing modules globally in each phase.

    Production:  TradingDeps.production()
    Tests:       make_test_deps() from tests/conftest.py
    """

    # ── Broker / execution ────────────────────────────────────────────────────
    trader: Any
    stock_scanner: Any

    # ── Market data ───────────────────────────────────────────────────────────
    market_data: Any
    insider_feed: Any
    av_sentiment: Any
    earnings_surprise: Any
    news_fetcher: Any
    options_scanner: Any
    sector_data: Any
    sentiment: Any

    # ── Analysis ──────────────────────────────────────────────────────────────
    ai_analyst: Any
    performance: Any

    # ── Notifications ─────────────────────────────────────────────────────────
    alerts: Any
    emailer: Any

    # ── Risk ──────────────────────────────────────────────────────────────────
    correlation: Any
    earnings_calendar: Any
    exit_optimiser: Any
    macro_calendar: Any
    position_sizer: Any
    risk_manager: Any

    # ── Utils ─────────────────────────────────────────────────────────────────
    audit_log: Any
    decision_log: Any
    portfolio_tracker: Any

    # ── Callables (function-level imports) ────────────────────────────────────
    get_latest_review: Any
    build_scan_universe: Any
    check_quote_gate: Any
    check_pre_trade: Any
    sanitize_headlines: Any
    validate_ai_response: Any
    get_day_summary: Any
    save_experiment_baseline: Any
    load_experiment_baseline: Any
    run_startup_health_check: Any

    # ── Config snapshot ───────────────────────────────────────────────────────
    run_config: RunConfig

    @classmethod
    def production(cls) -> TradingDeps:
        """Build from production modules. Call once at the start of each run."""
        from analysis import ai_analyst, performance
        from analysis.weekly_review import get_latest_review
        from data import (
            av_sentiment,
            earnings_surprise,
            insider_feed,
            market_data,
            news_fetcher,
            options_scanner,
            sector_data,
        )
        from data import sentiment as sentiment_module
        from execution import stock_scanner, trader
        from execution.quote_gate import check_quote_gate
        from execution.universe import build_scan_universe
        from notifications import alerts, emailer
        from risk import (
            correlation,
            earnings_calendar,
            exit_optimiser,
            macro_calendar,
            position_sizer,
            risk_manager,
        )
        from utils import audit_log, decision_log, portfolio_tracker
        from utils.health import run_startup_health_check
        from utils.portfolio_tracker import (
            get_day_summary,
            load_experiment_baseline,
            save_experiment_baseline,
        )
        from utils.validators import check_pre_trade, sanitize_headlines, validate_ai_response

        return cls(
            trader=trader,
            stock_scanner=stock_scanner,
            market_data=market_data,
            insider_feed=insider_feed,
            av_sentiment=av_sentiment,
            earnings_surprise=earnings_surprise,
            news_fetcher=news_fetcher,
            options_scanner=options_scanner,
            sector_data=sector_data,
            sentiment=sentiment_module,
            ai_analyst=ai_analyst,
            performance=performance,
            alerts=alerts,
            emailer=emailer,
            correlation=correlation,
            earnings_calendar=earnings_calendar,
            exit_optimiser=exit_optimiser,
            macro_calendar=macro_calendar,
            position_sizer=position_sizer,
            risk_manager=risk_manager,
            audit_log=audit_log,
            decision_log=decision_log,
            portfolio_tracker=portfolio_tracker,
            get_latest_review=get_latest_review,
            build_scan_universe=build_scan_universe,
            check_quote_gate=check_quote_gate,
            check_pre_trade=check_pre_trade,
            sanitize_headlines=sanitize_headlines,
            validate_ai_response=validate_ai_response,
            get_day_summary=get_day_summary,
            save_experiment_baseline=save_experiment_baseline,
            load_experiment_baseline=load_experiment_baseline,
            run_startup_health_check=run_startup_health_check,
            run_config=RunConfig.from_config(),
        )
