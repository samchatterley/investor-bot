import contextlib
import json
import os
from datetime import date, datetime

import pytz
from dotenv import load_dotenv

load_dotenv()


def today_et() -> date:
    """Return today's date in US Eastern time — use this for all market-date comparisons."""
    return datetime.now(pytz.timezone(MARKET_TIMEZONE)).date()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Explicit trading mode — set TRADING_MODE=live to enable live trading.
# If TRADING_MODE is set, the URL is validated against the expected Alpaca endpoint.
# If unset, falls back to substring detection for backward compatibility.
_TRADING_MODE = os.getenv("TRADING_MODE", "").lower()
if _TRADING_MODE == "paper":
    IS_PAPER = True
    _expected = "https://paper-api.alpaca.markets"
    if ALPACA_BASE_URL.rstrip("/") != _expected:
        raise ValueError(
            f"ALPACA_BASE_URL={ALPACA_BASE_URL!r} does not match TRADING_MODE=paper. "
            f"Expected {_expected!r}"
        )
elif _TRADING_MODE == "live":
    IS_PAPER = False
    _expected = "https://api.alpaca.markets"
    if ALPACA_BASE_URL.rstrip("/") != _expected:
        raise ValueError(
            f"ALPACA_BASE_URL={ALPACA_BASE_URL!r} does not match TRADING_MODE=live. "
            f"Expected {_expected!r}"
        )
elif _TRADING_MODE:
    raise ValueError(f"TRADING_MODE must be 'paper' or 'live', got {_TRADING_MODE!r}")
else:
    IS_PAPER = "paper" in ALPACA_BASE_URL

# Position sizing
MAX_POSITIONS = 5
MAX_POSITION_PCT = 0.45      # Deprecated — legacy Kelly cap; kept only for config.validate(). Superseded by MAX_POSITION_WEIGHT.
CASH_RESERVE_PCT = 0.10      # Always keep 10% as cash buffer

# Risk-budget sizing (replaces Kelly)
RISK_PER_TRADE_PCT = 0.0025  # 0.25% of equity risked per trade
MAX_POSITION_WEIGHT = 0.05   # 5% of portfolio per position (hard cap)

# Risk management
STOP_LOSS_PCT = 0.04         # 4% trailing stop (tighter than old fixed stop)
TAKE_PROFIT_PCT = 0.15       # 15% take profit target (let winners run a bit further)
TRAILING_STOP_PCT = 4.0      # percent trail below highest price (Alpaca native order)
KELLY_MULTIPLIER = 0.5       # half-Kelly — kept for research telemetry only

# AI decision threshold
MIN_CONFIDENCE = 7           # Min confidence score (1-10) to open a position

# Position hold limit — auto-exit after this many trading days
MAX_HOLD_DAYS = 3

# Per-signal hold limits override MAX_HOLD_DAYS for specific entry types.
# Momentum and trend trades need room to develop; mean-reversion and news
# catalysts play out faster and should be exited sooner.
SIGNAL_MAX_HOLD_DAYS: dict[str, int] = {
    "mean_reversion":     2,
    "rsi_oversold":       2,
    "news_catalyst":      2,
    "macd_crossover":     4,
    "momentum":           5,
    "trend_continuation": 5,
    "unknown":            3,  # conservative default
}

# Bear market filter — skip new buys when SPY drops more than this % in a single day
BEAR_MARKET_SPY_THRESHOLD = -1.5

# How many top movers to add to the daily scan universe
TOP_MOVERS_COUNT = 15

# Partial profit taking — sell half position when unrealised gain hits this %
PARTIAL_PROFIT_PCT = 8.0

# Earnings guard — exit positions with earnings within this many calendar days
EARNINGS_WARNING_DAYS = 2

# VIX thresholds for stop adjustment
VIX_HIGH = 25.0   # above this, widen stops

# Max positions per sector
MAX_SECTOR_POSITIONS = 2

# How many days of historical data to feed to Claude
LOOKBACK_DAYS = 30

# Claude model
CLAUDE_MODEL = "claude-sonnet-4-6"

# Market schedule (US Eastern Time)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 31      # Run 1 min after open
MARKET_TIMEZONE = "America/New_York"

# Stocks to scan - liquid names with fractional share support on Alpaca
STOCK_UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    # Semiconductors (high momentum sector)
    "AMD", "AVGO", "QCOM", "MU", "INTC", "TSM", "AMAT",
    # Software & growth tech
    "NFLX", "CRM", "ADBE", "UBER", "SHOP", "SNOW", "PLTR",
    # Fintech
    "PYPL", "SQ", "V", "MA",
    # Financials
    "JPM", "BAC", "GS", "MS",
    # Healthcare & pharma
    "LLY", "UNH", "JNJ", "ABBV", "MRK",
    # Energy
    "XOM", "CVX", "OXY",
    # Consumer discretionary
    "COST", "WMT", "HD", "MCD", "NKE", "SBUX",
    # Industrials
    "CAT", "DE", "GE",
    # Broad market & sector ETFs
    "SPY", "QQQ", "IWM", "XLK", "XLE", "XLF",
]

# Email notifications
EMAIL_FROM = os.getenv("EMAIL_FROM")           # Your Gmail address
EMAIL_TO = os.getenv("EMAIL_TO")               # Owner address — emergency alerts only
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")  # Gmail App Password (not your login password)
# Named recipient list for daily summary + weekly review emails.
# Format: "FirstName:email,FirstName:email,..."
# Example: "Sam:sam@gmail.com,Harri:harri@outlook.com,Jess:jess@gmail.com"
EMAIL_RECIPIENTS = os.getenv("EMAIL_RECIPIENTS", "")
# Legacy fallback — used if EMAIL_RECIPIENTS is not set
EMAIL_CC = os.getenv("EMAIL_CC", "")

# Log file path
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

# Pre-trade controls (MiFID II Article 17)
# Fat-finger guard: reject any single order above this USD value
MAX_SINGLE_ORDER_USD = 50000.0
# Runaway algorithm guard: halt new buys once this much notional has been deployed today
MAX_DAILY_NOTIONAL_USD = 150000.0

# Operations
# Kill switch creates this file; bot refuses to run while it exists.
# To resume: python main.py --clear-halt
HALT_FILE = os.path.join(LOG_DIR, ".HALTED")

# Max new buy orders placed in a single run — guards against runaway loops
MAX_ORDERS_PER_RUN = 3

# Minimum average daily volume — filters out illiquid stocks
MIN_VOLUME = 500_000

# Set to "I-ACCEPT-REAL-MONEY-RISK" to enable live trading without interactive prompt.
# The scheduler sets this in its environment; a human-initiated run requires typing it.
LIVE_CONFIRM = os.getenv("LIVE_CONFIRM", "")

# Path for runtime parameter overrides set by the weekly self-review.
# Values here take precedence over the defaults above at startup.
_RUNTIME_CONFIG_PATH = os.path.join(LOG_DIR, "runtime_config.json")

# Explicit allowlist of keys the weekly self-review may modify at runtime.
# Everything else — API keys, trading mode, stock universe — is immutable.
RUNTIME_OVERRIDE_KEYS: frozenset[str] = frozenset({
    "MIN_CONFIDENCE",
    "TRAILING_STOP_PCT",
    "MAX_HOLD_DAYS",
    "MAX_ORDERS_PER_RUN",
    "PARTIAL_PROFIT_PCT",
})

# (type, min_inclusive, max_inclusive) — applied only to runtime overrides.
# Tighter than the static validate() bounds to constrain AI self-modification.
RUNTIME_OVERRIDE_BOUNDS: dict[str, tuple] = {
    "MIN_CONFIDENCE":     (int,   7,    10),
    "TRAILING_STOP_PCT":  (float, 2.0,  10.0),
    "MAX_HOLD_DAYS":      (int,   1,    10),
    "MAX_ORDERS_PER_RUN": (int,   1,    5),
    "PARTIAL_PROFIT_PCT": (float, 3.0,  20.0),
}


def _load_runtime_overrides() -> None:
    """Apply allowlisted, bounds-checked parameter overrides from runtime_config.json.

    Unknown keys are rejected. Values outside the declared type or bounds are
    rejected. Every decision is written to the audit log for full observability.
    Audit failures never prevent startup.
    """
    try:
        with open(_RUNTIME_CONFIG_PATH) as f:
            overrides = json.load(f)
    except FileNotFoundError:
        return
    except json.JSONDecodeError as exc:
        import logging as _logging
        _logging.getLogger(__name__).warning(f"runtime_config.json is malformed: {exc}")
        return

    import sys
    module = sys.modules[__name__]

    for key, raw_value in overrides.items():
        if key not in RUNTIME_OVERRIDE_KEYS:
            with contextlib.suppress(Exception):
                _audit_config_event(
                    "CONFIG_OVERRIDE_REJECTED",
                    {"key": key, "value": raw_value, "reason": "not in allowlist"},
                )
            continue

        expected_type, min_val, max_val = RUNTIME_OVERRIDE_BOUNDS[key]

        try:
            coerced = expected_type(raw_value)
        except (TypeError, ValueError):
            with contextlib.suppress(Exception):
                _audit_config_event(
                    "CONFIG_OVERRIDE_REJECTED",
                    {"key": key, "value": raw_value,
                     "reason": f"cannot coerce to {expected_type.__name__}"},
                )
            continue

        if not (min_val <= coerced <= max_val):
            with contextlib.suppress(Exception):
                _audit_config_event(
                    "CONFIG_OVERRIDE_REJECTED",
                    {"key": key, "value": coerced,
                     "reason": f"out of bounds [{min_val}, {max_val}]"},
                )
            continue

        setattr(module, key, coerced)
        with contextlib.suppress(Exception):
            _audit_config_event("CONFIG_OVERRIDE_APPLIED", {"key": key, "value": coerced})


def _audit_config_event(event_type: str, payload: dict) -> None:
    """Lazy audit emit — avoids circular import (audit_log imports config)."""
    try:
        from utils import audit_log
        audit_log._write(event_type, payload)
    except Exception:
        pass


_load_runtime_overrides()


def validate():
    """Raise ValueError if any config value is outside its safe operating range."""
    errors = []
    if not (0 < MAX_POSITION_PCT <= 1.0):
        errors.append(f"MAX_POSITION_PCT={MAX_POSITION_PCT} must be between 0 and 1")
    if not (0 < CASH_RESERVE_PCT < 1.0):
        errors.append(f"CASH_RESERVE_PCT={CASH_RESERVE_PCT} must be between 0 and 1")
    if not (1 <= MIN_CONFIDENCE <= 10):
        errors.append(f"MIN_CONFIDENCE={MIN_CONFIDENCE} must be between 1 and 10")
    if TRAILING_STOP_PCT <= 0:
        errors.append(f"TRAILING_STOP_PCT={TRAILING_STOP_PCT} must be positive")
    if MAX_HOLD_DAYS < 1:
        errors.append(f"MAX_HOLD_DAYS={MAX_HOLD_DAYS} must be >= 1")
    if MAX_POSITIONS < 1:
        errors.append(f"MAX_POSITIONS={MAX_POSITIONS} must be >= 1")
    if MAX_SINGLE_ORDER_USD <= 0:
        errors.append("MAX_SINGLE_ORDER_USD must be positive")
    if MAX_DAILY_NOTIONAL_USD <= 0:
        errors.append("MAX_DAILY_NOTIONAL_USD must be positive")
    if errors:
        raise ValueError("Config errors:\n" + "\n".join(f"  - {e}" for e in errors))
