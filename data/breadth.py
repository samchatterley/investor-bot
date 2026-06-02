"""Market breadth indicators computed from universe price data.

Provides daily breadth snapshots used by:
  - regime_v2 (% above SMA inputs)
  - breadth_thrust signal (Zweig-style rapid expansion)
  - new_high_low_ratio gate
  - fear_greed_composite
  - ad_line trend

All functions accept an optional pre-loaded price_data dict to avoid
redundant downloads when called within the daily pipeline.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from config import LOG_DIR, STOCK_UNIVERSE, today_et

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")
_CACHE_PATH = os.path.join(LOG_DIR, "breadth_cache.json")


@dataclass(frozen=True)
class BreadthSnapshot:
    pct_above_sma50: float  # 0.0–1.0 — fraction of universe stocks above 50d SMA
    pct_above_sma200: float  # 0.0–1.0 — fraction above 200d SMA
    new_highs_52w: int  # count at new 52-week high today
    new_lows_52w: int  # count at new 52-week low today
    nh_nl_ratio: float  # new_highs / (new_lows + 1) to avoid div/0
    ad_line_5d_change: float  # 5-day change in cumulative advance-decline count
    breadth_thrust: bool  # True when Zweig condition triggered in last 10 days
    symbols_counted: int  # how many symbols had enough data


def _zero_snapshot() -> BreadthSnapshot:
    """Return a safe all-zeros snapshot for failure paths."""
    return BreadthSnapshot(
        pct_above_sma50=0.0,
        pct_above_sma200=0.0,
        new_highs_52w=0,
        new_lows_52w=0,
        nh_nl_ratio=0.0,
        ad_line_5d_change=0.0,
        breadth_thrust=False,
        symbols_counted=0,
    )


def is_breadth_thrust(
    pct_above_sma50_history: list[float],
    window: int = 10,
    from_threshold: float = 0.40,
    to_threshold: float = 0.60,
) -> bool:
    """Return True if pct_above_sma50 rose from below from_threshold to above
    to_threshold within any consecutive window of `window` bars.

    pct_above_sma50_history: ordered list of daily values (oldest first).
    Requires at least `window` values; returns False if list is shorter.
    """
    if len(pct_above_sma50_history) < window:
        return False
    for i in range(len(pct_above_sma50_history) - window + 1):
        chunk = pct_above_sma50_history[i : i + window]
        if chunk[0] < from_threshold and chunk[-1] > to_threshold:
            return True
    return False


def compute_breadth(
    price_data: dict[str, pd.DataFrame],
    min_bars_sma50: int = 55,
    min_bars_sma200: int = 205,
) -> BreadthSnapshot:
    """Compute breadth snapshot from pre-loaded price DataFrames.

    price_data: {symbol: DataFrame with 'Close', 'High', 'Low' columns, DatetimeIndex}
    Each DataFrame should have at least 260 bars for SMA200 to be meaningful.

    SMA50 computed from rolling(50).mean().
    SMA200 computed from rolling(200).mean().
    52-week high/low: max/min of last 252 bars.
    AD line: rolling sum of (advances - declines) over 5 days.
    breadth_thrust: True if pct_above_sma50 crossed from below 0.40 to above 0.60
        within any 10-trading-day window in the last 20 bars of history.
    """
    if not price_data:
        return _zero_snapshot()

    above_sma50_count = 0
    above_sma200_count = 0
    new_highs = 0
    new_lows = 0
    symbols_counted_sma50 = 0
    symbols_counted_sma200 = 0

    # For AD line: track daily (advances - declines) across the last 6 bars
    # We need 6 bars to compute a 5-day change in the cumulative sum
    ad_history_bars = 6

    # Collect per-bar advances/declines across all symbols (last 6 bars)
    per_bar_advances: list[int] = [0] * ad_history_bars
    per_bar_declines: list[int] = [0] * ad_history_bars
    ad_counted: int = 0

    # For breadth thrust: collect daily pct_above_sma50 history across last 20 bars
    thrust_window = 20
    per_bar_above_sma50: list[int] = [0] * thrust_window
    thrust_counted: int = 0

    for symbol, df in price_data.items():
        try:
            if df is None or df.empty or "Close" not in df.columns:
                continue

            close = df["Close"].dropna()
            n = len(close)

            # SMA50 check
            if n >= min_bars_sma50:
                sma50 = close.rolling(50).mean()
                latest_close = float(close.iloc[-1])
                latest_sma50 = float(sma50.iloc[-1])
                symbols_counted_sma50 += 1
                if latest_close > latest_sma50:
                    above_sma50_count += 1

            # SMA200 check
            if n >= min_bars_sma200:
                sma200 = close.rolling(200).mean()
                latest_close = float(close.iloc[-1])
                latest_sma200 = float(sma200.iloc[-1])
                symbols_counted_sma200 += 1
                if latest_close > latest_sma200:
                    above_sma200_count += 1

            # 52-week high/low (last 252 bars)
            if n >= 2:
                window_252 = close.iloc[-252:] if n >= 252 else close
                high_52w = float(window_252.max())
                low_52w = float(window_252.min())
                latest_close = float(close.iloc[-1])
                if latest_close >= high_52w:
                    new_highs += 1
                if latest_close <= low_52w:
                    new_lows += 1

            # AD line contributions: last ad_history_bars bars
            if n >= ad_history_bars + 1:
                recent = close.iloc[-(ad_history_bars + 1) :]
                daily_returns = recent.diff().dropna()
                ad_counted += 1
                for bar_idx in range(min(ad_history_bars, len(daily_returns))):
                    ret = float(daily_returns.iloc[bar_idx])
                    if ret > 0:
                        per_bar_advances[bar_idx] += 1
                    elif ret < 0:
                        per_bar_declines[bar_idx] += 1

            # Breadth thrust history: last thrust_window bars, need SMA50 for each
            if n >= min_bars_sma50 + thrust_window:
                sma50_series = close.rolling(50).mean()
                recent_close = close.iloc[-thrust_window:]
                recent_sma50 = sma50_series.iloc[-thrust_window:]
                thrust_counted += 1
                for bar_idx in range(thrust_window):
                    c = float(recent_close.iloc[bar_idx])
                    s = recent_sma50.iloc[bar_idx]
                    if pd.notna(s) and c > float(s):
                        per_bar_above_sma50[bar_idx] += 1

        except Exception as e:
            logger.debug(f"breadth: skipping {symbol}: {e}")
            continue

    symbols_counted = max(symbols_counted_sma50, symbols_counted_sma200)

    pct_above_sma50 = (
        above_sma50_count / symbols_counted_sma50 if symbols_counted_sma50 > 0 else 0.0
    )
    pct_above_sma200 = (
        above_sma200_count / symbols_counted_sma200 if symbols_counted_sma200 > 0 else 0.0
    )
    nh_nl_ratio = new_highs / (new_lows + 1)

    # AD line 5-day change: cumulative AD over last 5 bars vs 6-bars-ago
    if ad_counted > 0:
        ad_values = [per_bar_advances[i] - per_bar_declines[i] for i in range(ad_history_bars)]
        cumsum = [sum(ad_values[: i + 1]) for i in range(ad_history_bars)]
        ad_line_5d_change = float(cumsum[-1] - cumsum[0])
    else:
        ad_line_5d_change = 0.0

    # Breadth thrust check over last thrust_window bars
    if thrust_counted > 0:
        pct_history = [per_bar_above_sma50[i] / thrust_counted for i in range(thrust_window)]
        thrust = is_breadth_thrust(pct_history)
    else:
        thrust = False

    return BreadthSnapshot(
        pct_above_sma50=round(pct_above_sma50, 4),
        pct_above_sma200=round(pct_above_sma200, 4),
        new_highs_52w=new_highs,
        new_lows_52w=new_lows,
        nh_nl_ratio=round(nh_nl_ratio, 4),
        ad_line_5d_change=round(ad_line_5d_change, 4),
        breadth_thrust=thrust,
        symbols_counted=symbols_counted,
    )


def _load_breadth_cache() -> dict | None:
    """Return cached BreadthSnapshot dict for today (ET) or None on miss/stale/corrupt."""
    try:
        with open(_CACHE_PATH) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        cached_date = data.get("_date")
        if cached_date != today_et().isoformat():
            return None
        return data
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning(f"breadth cache read error: {e}")
        return None


def _save_breadth_cache(snapshot: BreadthSnapshot) -> None:
    """Save snapshot to disk. Silently swallows OSError."""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        payload = asdict(snapshot)
        payload["_date"] = today_et().isoformat()
        with open(_CACHE_PATH, "w") as f:
            json.dump(payload, f)
        logger.debug(f"breadth cache saved for {payload['_date']}")
    except OSError as e:
        logger.warning(f"breadth cache write error: {e}")


def get_breadth_snapshot(
    symbols: list[str] | None = None,
    price_data: dict[str, pd.DataFrame] | None = None,
    lookback_days: int = 260,
) -> BreadthSnapshot:
    """Fetch price data if not provided and compute breadth snapshot.

    If price_data is provided, uses it directly (avoids re-download).
    If symbols is None, uses config.STOCK_UNIVERSE.
    Returns a snapshot with symbols_counted=0 and all zeros/False on any failure.
    """
    # Check cache first (only when not given pre-loaded data)
    if price_data is None:
        cached = _load_breadth_cache()
        if cached is not None:
            try:
                cached_copy = {k: v for k, v in cached.items() if k != "_date"}
                snapshot = BreadthSnapshot(**cached_copy)
                logger.info("breadth: cache hit")
                return snapshot
            except Exception as e:
                logger.warning(f"breadth: cache deserialise error: {e}")

    try:
        if price_data is not None:
            data = price_data
        else:
            if symbols is None:
                symbols = list(STOCK_UNIVERSE)
            data = _download_price_data(symbols, lookback_days)

        snapshot = compute_breadth(data)

        if price_data is None:
            _save_breadth_cache(snapshot)

        return snapshot

    except Exception as e:
        logger.error(f"breadth: get_breadth_snapshot failed: {e}")
        return _zero_snapshot()


def _download_price_data(symbols: list[str], lookback_days: int) -> dict[str, pd.DataFrame]:
    """Download OHLCV history for symbols via yfinance batch download."""
    end = datetime.now(_ET)
    start = (end - timedelta(days=lookback_days + 30)).strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    try:
        raw = yf.download(
            tickers=symbols,
            start=start,
            end=end_str,
            auto_adjust=True,
            threads=False,
            progress=False,
        )
    except Exception as e:
        logger.warning(f"breadth: yfinance download failed: {e}")
        return {}

    if raw is None or raw.empty:
        return {}

    result: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        available = raw.columns.get_level_values(1).unique()
        for sym in symbols:
            if sym in available:
                try:
                    sym_df = raw.xs(sym, level=1, axis=1).dropna(how="all").copy()
                    if not sym_df.empty:
                        result[sym] = sym_df
                except KeyError:
                    pass
    elif len(symbols) == 1:
        result[symbols[0]] = raw.dropna(how="all").copy()

    logger.info(f"breadth: downloaded {len(result)}/{len(symbols)} symbols")
    return result
