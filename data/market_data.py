import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator, EMAIndicator
from ta.volatility import BollingerBands

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, MARKET_DATA_DIR
from data.fundamentals import get_fundamentals
from utils.alpaca_session import with_request_timeout
from utils.symbols import to_yf_symbol

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")
_MARKET_OPEN_TIME = (9, 30)  # hour, minute ET
_ORB_MINUTES = 30  # opening range window


def fetch_stock_data(
    symbol: str,
    days: int = 30,
    preloaded: dict | None = None,
    as_of: str | None = None,
) -> pd.DataFrame | None:
    """Fetch OHLCV data and compute technical indicators for a symbol.

    If ``preloaded`` is provided, uses the pre-downloaded DataFrame sliced to
    ``as_of`` date instead of a live yfinance call (for historical replay).
    """
    try:
        if preloaded is not None and symbol in preloaded:
            df = preloaded[symbol]
            if as_of is not None:
                df = df[df.index <= pd.Timestamp(as_of)].copy()
            else:
                df = df.copy()
            if df.empty or len(df) < 35:
                logger.warning(f"Insufficient preloaded data for {symbol} as_of {as_of}")
                return None
        else:
            end = datetime.now()
            # Weekly EMA21 needs 21 weeks (~150 calendar days); fetch enough for both daily and weekly indicators
            fetch_days = max(days + 150, 200)
            start = end - timedelta(days=fetch_days)

            ticker = yf.Ticker(to_yf_symbol(symbol))
            df = ticker.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))

            if df.empty or len(df) < 35:
                logger.warning(f"Insufficient data for {symbol}")
                return None

            last_date = df.index[-1].date()
            staleness = (datetime.now().date() - last_date).days
            if staleness > 4:
                logger.warning(
                    f"{symbol}: last data point {last_date} is {staleness} days old — skipping stale feed"
                )
                return None

            df = df.copy()

        close = df["Close"]
        volume = df["Volume"]

        # RSI
        df["rsi"] = RSIIndicator(close=close, window=14).rsi()

        # MACD
        macd = MACD(close=close)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()

        # EMAs
        df["ema9"] = EMAIndicator(close=close, window=9).ema_indicator()
        df["ema21"] = EMAIndicator(close=close, window=21).ema_indicator()

        # Bollinger Bands
        bb = BollingerBands(close=close, window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_pct"] = bb.bollinger_pband()  # 0=at lower band, 1=at upper band
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        # Squeeze: bandwidth in the lowest 20th percentile of the last 60 bars (3 months).
        # Using 60 bars rather than 20 prevents ~20% of days qualifying by construction;
        # a genuine squeeze requires compression relative to a longer-term baseline.
        bw_q20 = df["bb_bandwidth"].rolling(60, min_periods=20).quantile(0.2)
        df["bb_squeeze"] = df["bb_bandwidth"] <= bw_q20
        # Consecutive days in squeeze — evaluator requires ≥5 to filter transient dips.
        squeeze_int = df["bb_squeeze"].astype(int)
        # Rolling cumsum trick: count consecutive True streaks.
        cumsum = squeeze_int.cumsum()
        reset = cumsum - cumsum.where(~df["bb_squeeze"]).ffill().fillna(0)
        df["bb_squeeze_days"] = reset.astype(int)

        # Volume vs 20-day average
        df["avg_volume_20"] = volume.rolling(20).mean()
        df["vol_ratio"] = volume / df["avg_volume_20"]

        # Price momentum
        df["ret_1d"] = close.pct_change(1) * 100
        # Lottery/MAX gate (2026-07 workshop): a >=+10% single-day pop within the last 3 sessions.
        df["recent_lottery_pop"] = (df["ret_1d"] >= 10.0).astype(float).rolling(
            3, min_periods=1
        ).max() > 0
        df["ret_5d"] = close.pct_change(5) * 100
        df["ret_10d"] = close.pct_change(10) * 100
        df["ret_20d"] = close.pct_change(20) * 100

        # Historical volatility percentile: where today's 20-day HV sits in its 252-day range.
        # hv_rank=0.10 means current HV is in the bottom 10% → vol compression → squeeze likely.
        import math

        daily_returns = close.pct_change()
        df["hv_20d"] = daily_returns.rolling(20).std() * math.sqrt(252) * 100
        df["hv_rank"] = df["hv_20d"].rolling(252, min_periods=30).rank(pct=True)

        df = df.dropna(subset=["rsi", "macd"])

        # 52-week high (rolling 252-day max; min_periods=1 so short histories still work)
        df["high_52w"] = df["High"].rolling(252, min_periods=1).max()

        # ADX (14-period) — trend strength gate used by most momentum signals
        try:
            df["adx"] = (
                ADXIndicator(high=df["High"], low=df["Low"], close=close, window=14).adx().fillna(0)
            )
        except Exception:
            df["adx"] = 30.0  # assume trending if High/Low unavailable

        # Inside day: today's entire range is contained within the previous day's range
        df["is_inside_day"] = (df["High"] < df["High"].shift(1)) & (df["Low"] > df["Low"].shift(1))

        # failed_breakout: hit new 20-day high yesterday, failed back below it today.
        high_20d_lag2 = close.rolling(20, min_periods=10).max().shift(2)
        df["failed_breakout_flag"] = (
            (close.shift(1) > high_20d_lag2) & (close <= high_20d_lag2)
        ).fillna(False)

        # close_pct_of_range: where close sits in the day's High–Low range (0=low, 1=high).
        daily_range = df["High"] - df["Low"]
        df["close_pct_of_range"] = (
            (close - df["Low"]) / daily_range.where(daily_range > 0)
        ).fillna(0.5)

        # RSI divergence: price lower than 5 days ago but RSI recovering (bullish structural divergence)
        df["rsi_divergence"] = ((close < close.shift(5)) & (df["rsi"] > df["rsi"].shift(5))).fillna(
            False
        )

        # Weekly trend — resample daily to weekly, compute EMA9/EMA21/RSI on weekly candles
        try:
            weekly_close = close.resample("W-FRI").last().dropna()
            if len(weekly_close) >= 22:
                w_ema9 = EMAIndicator(close=weekly_close, window=9).ema_indicator().dropna()
                w_ema21 = EMAIndicator(close=weekly_close, window=21).ema_indicator().dropna()
                w_rsi = RSIIndicator(close=weekly_close, window=14).rsi().dropna()
                df["weekly_trend_up"] = bool(w_ema9.iloc[-1] > w_ema21.iloc[-1])
                df["weekly_rsi"] = round(float(w_rsi.iloc[-1]), 1)
            else:
                df["weekly_trend_up"] = True
                df["weekly_rsi"] = 50.0
        except Exception:
            df["weekly_trend_up"] = True
            df["weekly_rsi"] = 50.0

        # Rebind after dropna so Batch 1 ops never mix pre/post-dropna indices
        close = df["Close"]
        volume = df["Volume"]

        # ── Golden / Death Cross ──────────────────────────────────────────────
        df["sma50"] = close.rolling(50).mean()
        df["sma200"] = close.rolling(200).mean()
        _sma50_prev = df["sma50"].shift(1)
        _sma200_prev = df["sma200"].shift(1)
        df["golden_cross"] = ((_sma50_prev < _sma200_prev) & (df["sma50"] >= df["sma200"])).fillna(
            False
        )
        df["death_cross"] = ((_sma50_prev > _sma200_prev) & (df["sma50"] <= df["sma200"])).fillna(
            False
        )

        # ── On-Balance Volume and derivatives ─────────────────────────────────
        _price_dir = (close.diff() > 0).astype(int) - (close.diff() < 0).astype(int)
        df["obv"] = (volume * _price_dir).cumsum()
        df["obv_5d_slope"] = df["obv"].diff(5) / 5
        df["obv_20d_slope"] = df["obv"].diff(20) / 20
        _price_5d_chg = close.diff(5)
        df["obv_divergence_bull"] = ((_price_5d_chg < 0) & (df["obv_5d_slope"] > 0)).fillna(False)
        df["obv_divergence_bear"] = ((_price_5d_chg > 0) & (df["obv_5d_slope"] < 0)).fillna(False)
        df["obv_accelerating_up"] = (
            (df["obv_5d_slope"] > 0) & (df["obv_5d_slope"] > df["obv_20d_slope"])
        ).fillna(False)
        df["obv_accelerating_down"] = (
            (df["obv_5d_slope"] < 0) & (df["obv_5d_slope"] < df["obv_20d_slope"])
        ).fillna(False)

        # ── 20-day high / low proximity ───────────────────────────────────────
        df["high_20d"] = close.rolling(20, min_periods=10).max()
        df["low_20d"] = close.rolling(20, min_periods=10).min()
        _low_safe = df["low_20d"].where(df["low_20d"] > 0)
        _high_safe = df["high_20d"].where(df["high_20d"] > 0)
        df["near_20d_low"] = ((close - df["low_20d"]) / _low_safe < 0.02).fillna(False)
        df["near_20d_high"] = ((df["high_20d"] - close) / _high_safe < 0.02).fillna(False)

        # ── Candle patterns ───────────────────────────────────────────────────
        _body = close - df["Open"]
        _body_abs = _body.abs()
        _candle_top = close.where(_body >= 0, df["Open"])
        _candle_bot = close.where(_body < 0, df["Open"])
        _upper_shadow = df["High"] - _candle_top
        _lower_shadow = _candle_bot - df["Low"]
        df["hammer"] = (
            (_body_abs > 0) & (_lower_shadow >= 2 * _body_abs) & (_upper_shadow <= 0.3 * _body_abs)
        ).fillna(False)
        df["bullish_engulf"] = (
            (_body >= 0)
            & (close.shift(1) < df["Open"].shift(1))
            & (close > df["Open"].shift(1))
            & (df["Open"] < close.shift(1))
        ).fillna(False)
        df["shooting_star"] = (
            (_body_abs > 0) & (_upper_shadow >= 2 * _body_abs) & (_lower_shadow <= 0.3 * _body_abs)
        ).fillna(False)
        df["bearish_engulf"] = (
            (_body < 0)
            & (close.shift(1) > df["Open"].shift(1))
            & (close < df["Open"].shift(1))
            & (df["Open"] > close.shift(1))
        ).fillna(False)

        # ── High-volume streak ────────────────────────────────────────────────
        _is_climax = df["vol_ratio"] > 2.5
        _climax_cs = _is_climax.astype(int).cumsum()
        _streak = _climax_cs - _climax_cs.where(~_is_climax).ffill().fillna(0)
        df["high_vol_streak"] = _streak.astype(int)

        # ── Spread proxy (execution-cost gate) ───────────────────────────────
        if "High" in df.columns and "Low" in df.columns:
            _midpoint = (df["High"] + df["Low"]) / 2
            _spread_raw = (df["High"] - df["Low"]) / _midpoint.where(_midpoint > 0)
            df["spread_proxy_20d"] = _spread_raw.rolling(20, min_periods=10).mean().fillna(0.0)
        else:  # pragma: no cover
            df["spread_proxy_20d"] = 0.0

        # Return only the requested number of most recent days
        return df.tail(days) if len(df) >= 2 else None

    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None


def summarise_for_ai(symbol: str, df: pd.DataFrame, is_preloaded: bool = False) -> dict:
    """Extract a compact summary of the latest technical snapshot."""
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    _idx = df.index[-1]
    try:
        _bar_date: str | None = _idx.strftime("%Y-%m-%d")
        _bar_is_final: bool | None = _idx.date() < date.today()
    except AttributeError:
        _bar_date = None
        _bar_is_final = None

    return {
        "symbol": symbol,
        "bar_date": _bar_date,
        "bar_is_final": _bar_is_final,
        "data_source": "preloaded" if is_preloaded else "live",
        "current_price": round(float(latest["Close"]), 2),
        "ret_1d_pct": round(float(latest["ret_1d"]), 2),
        "recent_lottery_pop": bool(latest.get("recent_lottery_pop", False)),
        "ret_5d_pct": round(float(latest["ret_5d"]), 2),
        "ret_10d_pct": round(float(latest["ret_10d"]), 2),
        "ret_20d_pct": round(float(_r20), 2)
        if (_r20 := latest.get("ret_20d")) is not None and pd.notna(_r20)
        else 0.0,
        "rsi_14": round(float(latest["rsi"]), 1),
        "macd_diff": round(float(latest["macd_diff"]), 4),
        "macd_crossed_up": bool(latest["macd_diff"] > 0 and prev["macd_diff"] <= 0),
        "macd_crossed_down": bool(latest["macd_diff"] < 0 and prev["macd_diff"] >= 0),
        "ema9_above_ema21": bool(latest["ema9"] > latest["ema21"]),
        "bb_pct": round(float(latest["bb_pct"]), 2),  # 0=oversold zone, 1=overbought zone
        "vol_ratio": round(float(latest["vol_ratio"]), 2),  # >1.5 = high volume
        "avg_volume": int(float(_av))
        if (_av := latest.get("avg_volume_20")) is not None and not pd.isna(_av)
        else 0,
        "price_vs_ema9_pct": round((float(latest["Close"]) / float(latest["ema9"]) - 1) * 100, 2),
        "price_vs_ema21_pct": round((float(latest["Close"]) / float(latest["ema21"]) - 1) * 100, 2),
        "weekly_trend_up": bool(latest.get("weekly_trend_up", True)),
        "weekly_rsi": float(latest.get("weekly_rsi", 50.0)),
        "bb_squeeze": bool(latest.get("bb_squeeze", False))
        if pd.notna(latest.get("bb_squeeze", False))
        else False,
        "bb_squeeze_days": int(latest.get("bb_squeeze_days", 0))
        if pd.notna(latest.get("bb_squeeze_days", 0))
        else 0,
        "is_inside_day": bool(latest.get("is_inside_day", False))
        if pd.notna(latest.get("is_inside_day", False))
        else False,
        "price_vs_52w_high_pct": round((float(latest["Close"]) / float(_h52w) - 1) * 100, 2)
        if (_h52w := latest.get("high_52w")) is not None and pd.notna(_h52w)
        else 0.0,
        "hv_rank": round(float(_hvr), 2)
        if (_hvr := latest.get("hv_rank")) is not None and pd.notna(_hvr)
        else 1.0,
        "adx": round(float(_adx), 1)
        if (_adx := latest.get("adx")) is not None and pd.notna(_adx)
        else 30.0,
        "rsi_divergence": bool(latest.get("rsi_divergence", False))
        if pd.notna(latest.get("rsi_divergence", False))
        else False,
        "failed_breakout_flag": bool(latest.get("failed_breakout_flag", False))
        if pd.notna(latest.get("failed_breakout_flag", False))
        else False,
        "close_pct_of_range": round(float(_cpr), 2)
        if (_cpr := latest.get("close_pct_of_range")) is not None and pd.notna(_cpr)
        else 0.5,
        # ── Batch 1 OHLCV signal fields ───────────────────────────────────────
        "golden_cross": bool(latest.get("golden_cross", False))
        if pd.notna(latest.get("golden_cross", False))
        else False,
        "death_cross": bool(latest.get("death_cross", False))
        if pd.notna(latest.get("death_cross", False))
        else False,
        "obv_divergence_bull": bool(latest.get("obv_divergence_bull", False))
        if pd.notna(latest.get("obv_divergence_bull", False))
        else False,
        "obv_divergence_bear": bool(latest.get("obv_divergence_bear", False))
        if pd.notna(latest.get("obv_divergence_bear", False))
        else False,
        "obv_accelerating_up": bool(latest.get("obv_accelerating_up", False))
        if pd.notna(latest.get("obv_accelerating_up", False))
        else False,
        "obv_accelerating_down": bool(latest.get("obv_accelerating_down", False))
        if pd.notna(latest.get("obv_accelerating_down", False))
        else False,
        "near_20d_low": bool(latest.get("near_20d_low", False))
        if pd.notna(latest.get("near_20d_low", False))
        else False,
        "near_20d_high": bool(latest.get("near_20d_high", False))
        if pd.notna(latest.get("near_20d_high", False))
        else False,
        "hammer": bool(latest.get("hammer", False))
        if pd.notna(latest.get("hammer", False))
        else False,
        "bullish_engulf": bool(latest.get("bullish_engulf", False))
        if pd.notna(latest.get("bullish_engulf", False))
        else False,
        "shooting_star": bool(latest.get("shooting_star", False))
        if pd.notna(latest.get("shooting_star", False))
        else False,
        "bearish_engulf": bool(latest.get("bearish_engulf", False))
        if pd.notna(latest.get("bearish_engulf", False))
        else False,
        "high_vol_streak": int(latest.get("high_vol_streak", 0))
        if pd.notna(latest.get("high_vol_streak", 0))
        else 0,
        # ── Batch 2 OHLCV signal fields ───────────────────────────────────────
        "spread_proxy_20d": round(float(latest.get("spread_proxy_20d", 0.0)), 5)
        if pd.notna(latest.get("spread_proxy_20d", 0.0))
        else 0.0,
        # ── Batch 3: calendar context ─────────────────────────────────────────
        "calendar_month": date.today().month,
    }


def _apply_sector_ret5d(snapshots: list[dict], min_members: int = 5) -> None:
    """Set sector_ret_5d_pct (equal-weight sector-peer 5d return, %) on each snapshot in place.

    Feeds the residual_reversal sector conjunct (v1.144): a -7% drop must clear the threshold vs the
    name's own sector too, else it's sector beta (which continues), not idiosyncratic flow (which
    reverts). Sectors with fewer than ``min_members`` priced members are skipped (field absent →
    the evaluator fails open to the spy-only construction). Unknown sectors are skipped.
    """
    from data.sector_data import get_sector

    groups: dict[str, list[float]] = {}
    sec_of: dict[int, str] = {}
    for idx, s in enumerate(snapshots):
        r5 = s.get("ret_5d_pct")
        if r5 is None:
            continue
        sec = get_sector(s.get("symbol", ""))
        if not sec or sec == "Unknown":
            continue
        sec_of[idx] = sec
        groups.setdefault(sec, []).append(float(r5))
    means = {sec: sum(v) / len(v) for sec, v in groups.items() if len(v) >= min_members}
    for idx, sec in sec_of.items():
        if sec in means:
            snapshots[idx]["sector_ret_5d_pct"] = round(means[sec], 2)


def compute_amihud_illiquidity(df: pd.DataFrame, lookback: int = 20) -> float:
    """Return the Amihud (2002) illiquidity ratio averaged over the last *lookback* bars.

    illiq = mean( |daily_return| / (close * volume) )

    Higher = less liquid.  Returns 0.0 when data is insufficient or all dollar-volume
    is zero (e.g. zero-volume periods in backtest pre-loaded data).
    """
    try:
        tail = df.tail(lookback)
        if len(tail) < 5:
            return 0.0
        ret = tail["Close"].pct_change().abs()
        dollar_vol = tail["Close"] * tail["Volume"]
        ratio = ret / dollar_vol.replace(0, float("nan"))
        mean_val = float(ratio.mean(skipna=True))
        if mean_val != mean_val:  # NaN guard
            return 0.0
        return mean_val
    except Exception:
        return 0.0


def get_vix() -> float | None:
    """Return the latest VIX close, or None if the feed is degraded (logged, not silent)."""
    try:
        hist = yf.Ticker("^VIX").history(period="3d")
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 1)
    except Exception as e:
        logger.warning("VIX fetch failed — regime stress/HV triggers degraded this cycle: %s", e)
        return None
    logger.warning(
        "VIX unavailable (empty history) — regime stress/HV triggers degraded this cycle"
    )
    return None


def get_index_price(symbol: str = "SPY") -> float | None:
    """Return the latest close price for an index ETF (used to size the index regime hedge).

    Returns None on any fetch failure so the caller can skip the hedge rather than crash.
    """
    try:
        hist = yf.Ticker(symbol).history(period="5d")
        if len(hist) >= 1:
            return round(float(hist["Close"].iloc[-1]), 4)
    except Exception:
        pass
    return None


def get_spy_5d_return() -> float | None:
    """Return SPY's 5-day return % for relative strength calculation.

    On any failure returns None BUT logs it — a silent None here suppresses residual_reversal
    and rs_leader for the whole cycle (both gate on spy_ret_5d is not None).
    """
    try:
        hist = yf.Ticker("SPY").history(period="15d")
        if len(hist) >= 6:
            return round(
                (float(hist["Close"].iloc[-1]) / float(hist["Close"].iloc[-6]) - 1) * 100, 2
            )
    except Exception as e:
        logger.warning(
            "SPY 5d return fetch failed — RS/reversal context degraded this cycle: %s", e
        )
        return None
    logger.warning("SPY 5d return unavailable — RS/reversal context degraded this cycle")
    return None


def get_spy_10d_return() -> float | None:
    """Return SPY's 10-day return % for relative strength calculation (logs on degradation)."""
    try:
        hist = yf.Ticker("SPY").history(period="25d")
        if len(hist) >= 11:
            return round(
                (float(hist["Close"].iloc[-1]) / float(hist["Close"].iloc[-11]) - 1) * 100, 2
            )
    except Exception as e:
        logger.warning("SPY 10d return fetch failed — RS context degraded this cycle: %s", e)
        return None
    logger.warning("SPY 10d return unavailable — RS context degraded this cycle")
    return None


def get_spy_20d_return() -> float | None:
    """Return SPY's 20-day return % for relative strength calculation."""
    try:
        hist = yf.Ticker("SPY").history(period="35d")
        if len(hist) >= 21:
            return round(
                (float(hist["Close"].iloc[-1]) / float(hist["Close"].iloc[-21]) - 1) * 100, 2
            )
    except Exception:
        pass
    return None


def _spy_return_from_preloaded(preloaded: dict, as_of: str, lookback: int) -> float | None:
    """Compute SPY N-day return from preloaded data up to as_of date."""
    spy_df = preloaded.get("SPY")
    if spy_df is None:
        return None
    try:
        sliced = spy_df[spy_df.index <= pd.Timestamp(as_of)]["Close"].dropna()
        if len(sliced) < lookback + 1:
            return None
        return round((float(sliced.iloc[-1]) / float(sliced.iloc[-(lookback + 1)]) - 1) * 100, 2)
    except Exception:
        return None


def _bulk_cache_path() -> str:
    today = datetime.now(_ET).date().isoformat()
    return os.path.join(MARKET_DATA_DIR, f"market_data_{today}.pkl")


def _load_bulk_cache() -> dict[str, pd.DataFrame]:
    path = _bulk_cache_path()
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            logger.info(f"Bulk cache hit: {len(data)} symbols from {os.path.basename(path)}")
            return data
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"Bulk cache read error: {e}")
    return {}


_BULK_CACHE_KEEP_DAYS = 3  # retain this many days of market_data_*.pkl; older ones are auto-pruned


def _prune_old_bulk_caches(keep_days: int = _BULK_CACHE_KEEP_DAYS) -> int:
    """Delete market_data_*.pkl caches older than keep_days (by filename date); return count pruned.

    These regenerable daily caches accumulate ~4-5 MB/day; without pruning logs/ grows unbounded
    (audit: logs/ cleanup). The most recent keep_days are retained.
    """
    cutoff = (datetime.now(_ET).date() - timedelta(days=keep_days)).isoformat()
    pruned = 0
    try:
        names = os.listdir(MARKET_DATA_DIR)
    except OSError:  # pragma: no cover — log dir missing
        return 0
    for name in names:
        if not (name.startswith("market_data_") and name.endswith(".pkl")):
            continue
        datestr = name[len("market_data_") : -len(".pkl")]
        if len(datestr) == 10 and datestr < cutoff:
            try:
                os.remove(os.path.join(MARKET_DATA_DIR, name))
                pruned += 1
            except OSError:  # pragma: no cover — file vanished between listdir and remove
                pass
    if pruned:
        logger.info(f"Pruned {pruned} market_data cache(s) older than {keep_days}d")
    return pruned


def _save_bulk_cache(data: dict[str, pd.DataFrame]) -> None:
    path = _bulk_cache_path()
    try:
        os.makedirs(MARKET_DATA_DIR, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Bulk cache saved: {len(data)} symbols → {os.path.basename(path)}")
        _prune_old_bulk_caches()
    except Exception as e:
        logger.warning(f"Bulk cache write error: {e}")


def migrate_bulk_caches_to_subdir() -> int:
    """One-time: move legacy logs/market_data_*.pkl (root) into logs/market_data/; return count moved.

    Lets an existing install adopt the new foldering on restart without re-downloading today's cache.
    """
    from config import LOG_DIR

    moved = 0
    try:
        names = os.listdir(LOG_DIR)
    except OSError:  # pragma: no cover — log dir missing
        return 0
    for name in names:
        if not (name.startswith("market_data_") and name.endswith(".pkl")):
            continue
        src = os.path.join(LOG_DIR, name)
        dst = os.path.join(MARKET_DATA_DIR, name)
        if os.path.isfile(src) and not os.path.exists(dst):
            try:
                os.makedirs(MARKET_DATA_DIR, exist_ok=True)
                os.replace(src, dst)
                moved += 1
            except OSError:  # pragma: no cover — file vanished / cross-device move
                pass
    if moved:
        logger.info(f"Migrated {moved} legacy market_data cache(s) into {MARKET_DATA_DIR}")
    return moved


def _download_symbols(symbols: list[str], fetch_days: int) -> dict[str, pd.DataFrame]:
    """Raw yfinance batch download for a list of symbols."""
    end = datetime.now()
    start = (end - timedelta(days=fetch_days)).strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    # Query yfinance in its own ticker convention (BRK.B -> BRK-B) but map results back to the
    # original symbols so the rest of the system keeps using the dot form.
    yf_to_orig = {to_yf_symbol(s): s for s in symbols}
    try:
        raw = yf.download(
            tickers=list(yf_to_orig),
            start=start,
            end=end_str,
            auto_adjust=True,
            threads=False,
            progress=False,
        )
    except Exception as e:
        logger.warning(f"Bulk yfinance download failed: {e}")
        return {}
    if raw is None or raw.empty:
        return {}
    result: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        available = raw.columns.get_level_values(1).unique()
        for yf_sym, orig in yf_to_orig.items():
            if yf_sym in available:
                try:
                    sym_df = raw.xs(yf_sym, level=1, axis=1).dropna(how="all").copy()
                    if not sym_df.empty:
                        result[orig] = sym_df
                except KeyError:
                    pass
    elif len(symbols) == 1:
        result[symbols[0]] = raw.dropna(how="all").copy()
    return result


def _bulk_download(symbols: list[str], fetch_days: int) -> dict[str, pd.DataFrame]:
    """Download OHLCV for all symbols, using a same-day disk cache.

    The first call each ET calendar day downloads everything and saves to
    logs/market_data_YYYY-MM-DD.pkl.  Subsequent calls (later trading
    windows, or a full re-run) load from cache and only download symbols
    not already present (e.g. dynamic top-movers added after the prefetch).
    This eliminates the 60-90 min repeat download on every intraday trigger.
    """
    cache = _load_bulk_cache()
    missing = [s for s in symbols if s not in cache]

    if not missing:
        return {s: cache[s] for s in symbols if s in cache}

    logger.info(
        f"Downloading {len(missing)} symbols (cache has {len(cache)}, total requested {len(symbols)})"
    )
    fresh = _download_symbols(missing, fetch_days)

    if fresh:
        cache.update(fresh)
        _save_bulk_cache(cache)
    elif not cache:
        logger.warning("Bulk download returned no data — per-symbol fallback active")

    return {s: cache[s] for s in symbols if s in cache}


def prefetch_market_data(symbols: list[str]) -> None:
    """Warm the same-day bulk cache before market open.

    Called by the scheduler at 09:00 ET so that the 09:31 open_sells and
    10:00 open_buys runs load from disk instead of re-downloading.  Safe to
    call multiple times — subsequent calls are no-ops if cache is already warm.
    """
    fetch_days = max(config_lookback() + 150, 200)
    logger.info(f"Pre-market prefetch: {len(symbols)} symbols, fetch_days={fetch_days}")
    cached = _load_bulk_cache()
    missing = [s for s in symbols if s not in cached]
    if not missing:
        logger.info("Prefetch: cache already warm, nothing to download")
        return
    fresh = _download_symbols(missing, fetch_days)
    if fresh:
        cached.update(fresh)
        _save_bulk_cache(cached)
        logger.info(f"Prefetch complete: {len(cached)} symbols cached")
    else:
        logger.warning("Prefetch: download returned no data")


def config_lookback() -> int:
    """Return LOOKBACK_DAYS from config without a circular import."""
    from config import LOOKBACK_DAYS  # local import — avoids module-level circular dep

    return LOOKBACK_DAYS


def get_market_snapshots(
    symbols: list[str],
    days: int = 30,
    preloaded: dict | None = None,
    as_of: str | None = None,
) -> list[dict]:
    """Fetch and summarise data for all symbols in parallel."""
    live_bulk: dict[str, pd.DataFrame] | None = None
    fundamentals: dict[str, dict] = {}
    _fundam_fields: dict[str, dict] = {}
    _aaii_snap = None
    _macro_10y: float | None = None
    _analyst_revisions: dict[str, dict] = {}
    _lockup_flags: dict[str, dict] = {}
    if preloaded is not None and as_of is not None:
        spy_5d = _spy_return_from_preloaded(preloaded, as_of, 5)
        spy_10d = _spy_return_from_preloaded(preloaded, as_of, 10)
        spy_20d = _spy_return_from_preloaded(preloaded, as_of, 20)
    else:
        spy_5d = get_spy_5d_return()
        spy_10d = get_spy_10d_return()
        spy_20d = get_spy_20d_return()
        # Single bulk download replaces 75+ parallel Ticker.history() calls
        fetch_days = max(days + 150, 200)
        live_bulk = _bulk_download(symbols, fetch_days)
        if live_bulk:
            logger.info(f"Bulk download: {len(live_bulk)}/{len(symbols)} symbols fetched")
        else:
            logger.warning("Bulk download returned no data — per-symbol fallback active")
        # Fundamentals from FMP — cached 24h, negligible cost after first fill
        fundamentals = get_fundamentals(symbols)

        # Deep fundamentals from yfinance — weekly cache, injected per symbol
        try:
            from data.fundamental_cache import refresh_fundamental_cache

            refresh_fundamental_cache(symbols)
            from data.fundamental_cache import (
                get_accruals_ratio,
                get_altman_z,
                get_fcf_yield,
                get_forward_pe,
                get_gross_margin_current,
                get_gross_margin_trend,
                get_piotroski_f,
            )

            for _sym in symbols:
                _fundam_fields[_sym] = {
                    k: v
                    for k, v in {
                        "piotroski_f": get_piotroski_f(_sym),
                        "altman_z": get_altman_z(_sym),
                        "fcf_yield": get_fcf_yield(_sym),
                        "gross_margin_current": get_gross_margin_current(_sym),
                        "gross_margin_trend": get_gross_margin_trend(_sym),
                        "accruals_ratio": get_accruals_ratio(_sym),
                        "forward_pe": get_forward_pe(_sym),
                    }.items()
                    if v is not None
                }
        except Exception as _exc:
            logger.warning(f"fundamental_cache injection failed: {_exc}")

        # AAII sentiment + 10y yield — market-wide signals injected into all snapshots. AAII comes
        # from sentiment_client (aaii.com survey); it is NOT a FRED series. 10y yield is FRED.
        try:
            from data.fred_client import get_10y_yield
            from data.sentiment_client import get_aaii_sentiment

            _aaii_snap = get_aaii_sentiment()
            _macro_10y = get_10y_yield()
        except Exception as _exc:
            logger.warning(f"AAII/10y injection failed: {_exc}")

        # Analyst revision signals — per-symbol
        try:
            from data.analyst_revisions import get_analyst_revisions

            _analyst_revisions = get_analyst_revisions(symbols)
        except Exception as _exc:
            logger.warning(f"analyst_revisions injection failed: {_exc}")

        # Lockup expiry flags — per-symbol
        try:
            from data.lockup_calendar import get_lockup_expiry_flags

            _lockup_flags = get_lockup_expiry_flags(symbols)
        except Exception as _exc:
            logger.warning(f"lockup_calendar injection failed: {_exc}")

    def _fetch_one(sym: str):
        # Backtest replay uses preloaded; live runs use bulk cache (fallback: per-symbol fetch)
        data_src = preloaded if preloaded is not None else live_bulk
        df = fetch_stock_data(sym, days, preloaded=data_src, as_of=as_of)
        if df is None:
            return None
        # is_preloaded=True only for backtest replay (preloaded + as_of both provided)
        is_hist = preloaded is not None and as_of is not None
        snap = summarise_for_ai(sym, df, is_preloaded=is_hist)
        if spy_5d is not None:
            snap["rel_strength_5d"] = round(snap["ret_5d_pct"] - spy_5d, 2)
        if spy_10d is not None:
            snap["rel_strength_10d"] = round(snap["ret_10d_pct"] - spy_10d, 2)
        if spy_20d is not None and snap.get("ret_20d_pct") is not None:
            snap["rel_strength_20d"] = round(snap["ret_20d_pct"] - spy_20d, 2)
        if sym in fundamentals:
            snap.update(fundamentals[sym])
        if sym in _fundam_fields:
            snap.update(_fundam_fields[sym])
        if _aaii_snap is not None:
            snap["aaii_extreme_fear"] = _aaii_snap.extreme_bearish
            snap["aaii_excessive_bulls"] = _aaii_snap.extreme_bullish
            snap["aaii_bears_pct"] = round(_aaii_snap.bearish_pct * 100, 1)
        if _macro_10y is not None:
            snap["macro_10y_yield"] = _macro_10y
        if sym in _analyst_revisions:
            snap.update(_analyst_revisions[sym])
        if sym in _lockup_flags:
            snap.update(_lockup_flags[sym])
        snap["amihud_illiquidity"] = compute_amihud_illiquidity(df)
        return snap

    snapshots = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for snap in executor.map(_fetch_one, symbols):
            if snap is not None:
                snapshots.append(snap)

    _apply_sector_ret5d(snapshots)

    # Cross-sectional RS rank: percentile within this universe (100 = top).
    # Requires rel_strength_20d on at least 4 snapshots to be meaningful.
    rs20_pairs = [
        (i, s["rel_strength_20d"]) for i, s in enumerate(snapshots) if "rel_strength_20d" in s
    ]
    if len(rs20_pairs) >= 4:
        n = len(rs20_pairs)
        sorted_scores = sorted(v for _, v in rs20_pairs)
        for idx, score in rs20_pairs:
            snapshots[idx]["rs_rank_pct"] = round(
                sum(1 for v in sorted_scores if v < score) / n * 100, 1
            )

    # Cross-sectional Amihud illiquidity rank: flag top 10% as illiquid.
    amihud_pairs = [
        (i, s["amihud_illiquidity"])
        for i, s in enumerate(snapshots)
        if s.get("amihud_illiquidity", 0.0) > 0
    ]
    if len(amihud_pairs) >= 10:
        threshold = sorted(v for _, v in amihud_pairs)[int(len(amihud_pairs) * 0.90)]
        for idx, val in amihud_pairs:
            snapshots[idx]["amihud_illiquid"] = val >= threshold
    else:
        for s in snapshots:
            s["amihud_illiquid"] = False

    # Breadth + NHL ratio injection (live pipeline only — backtest uses engine-level dict)
    if live_bulk is not None:
        try:
            from data.breadth import get_breadth_snapshot

            _bsnapshot = get_breadth_snapshot(price_data=live_bulk)
            _bt_flag = bool(_bsnapshot.breadth_thrust)
            _bt_count = int(_bsnapshot.symbols_counted)
            _nhl = round(float(_bsnapshot.nh_nl_ratio), 4)
            for s in snapshots:
                s["breadth_thrust"] = _bt_flag
                s["breadth_symbols_counted"] = _bt_count
                s["nhl_ratio"] = _nhl
        except Exception as exc:
            logger.warning(f"breadth_thrust injection failed: {exc}")
            for s in snapshots:
                s.setdefault("breadth_thrust", False)
                s.setdefault("breadth_symbols_counted", 0)
                s.setdefault("nhl_ratio", 1.0)

    # Sector correlation injection (live pipeline only)
    if live_bulk is not None:
        try:
            from data.sector_correlation import compute_stock_sector_corr
            from data.sector_data import SECTOR_ETFS, get_sector_etf

            # Ensure sector ETF price data is available in bulk cache
            _all_etfs = list(set(SECTOR_ETFS.values()))
            _etf_data = _bulk_download(_all_etfs, max(days + 150, 200))
            if _etf_data:
                live_bulk.update(_etf_data)

            for s in snapshots:
                _sym = s["symbol"]
                _etf = get_sector_etf(_sym)
                if _etf:
                    _corr = compute_stock_sector_corr(_sym, _etf, price_data=live_bulk)
                    if _corr is not None:
                        s["sector_correlation_20d"] = _corr
        except Exception as exc:
            logger.warning(f"sector correlation injection failed: {exc}")

    return snapshots


def get_intraday_data(symbols: list[str]) -> dict[str, dict]:
    """Fetch Alpaca minute bars from market open and compute intraday metrics.

    Returns a dict keyed by symbol.  Each value contains:
      gap_pct              — (open - prev_close) / prev_close * 100
      intraday_change_pct  — (current - open) / open * 100
      intraday_vol_ratio   — cumulative volume / expected volume at this time of day
      price_above_vwap     — bool: current price > VWAP
      pct_vs_vwap          — (current - vwap) / vwap * 100
      orb_high / orb_low   — opening range (first _ORB_MINUTES minutes) high/low
      orb_breakout_up      — current price crossed above orb_high with vol confirmation
      orb_breakout_down    — current price crossed below orb_low
      intraday_rsi         — RSI-14 computed on 5-minute bars (None if < 15 bars)

    Returns an empty dict for any symbol where Alpaca data is unavailable.
    Failures are per-symbol and silent so they never block the main pipeline.
    """
    if not symbols:
        return {}

    now_et = datetime.now(_ET)
    market_open_et = now_et.replace(
        hour=_MARKET_OPEN_TIME[0], minute=_MARKET_OPEN_TIME[1], second=0, microsecond=0
    )

    # Nothing to compute before market open
    if now_et < market_open_et:
        return {}

    # Start 30 min before open to capture the last pre-market bar for gap calculation
    start_utc = (market_open_et - timedelta(minutes=30)).astimezone(UTC)
    end_utc = now_et.astimezone(UTC)

    try:
        data_client = with_request_timeout(
            StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        )
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=start_utc,
            end=end_utc,
            feed="iex",  # type: ignore[arg-type]
        )
        bars_response = data_client.get_stock_bars(req)
    except Exception as e:  # pragma: no cover
        logger.warning(f"Intraday data fetch failed: {e}")
        return {}

    bar_data = bars_response.data  # type: ignore[union-attr]
    result: dict[str, dict] = {}

    for sym in symbols:
        try:
            sym_bars = bar_data.get(sym)
            if not sym_bars:  # pragma: no cover
                continue

            df = pd.DataFrame(
                [
                    {
                        "t": b.timestamp,
                        "open": float(b.open),
                        "high": float(b.high),
                        "low": float(b.low),
                        "close": float(b.close),
                        "volume": float(b.volume),
                    }
                    for b in sym_bars
                ]
            )
            df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert(_ET)
            df = df.sort_values("t").reset_index(drop=True)

            if df.empty:
                continue

            # Previous close = last bar before today's market open
            pre_market = df[df["t"] < market_open_et]
            today_bars = df[df["t"] >= market_open_et]

            if today_bars.empty:
                continue

            prev_close = float(pre_market["close"].iloc[-1]) if not pre_market.empty else None
            day_open = float(today_bars["open"].iloc[0])
            current_price = float(today_bars["close"].iloc[-1])

            # Gap
            gap_pct = round((day_open / prev_close - 1) * 100, 2) if prev_close else None

            # Premarket gap quality: has the opening gap retraced >50% within the first 5 bars?
            # Only evaluated for upside gaps (gap_and_go is a long signal requiring gap ≥2%).
            premarket_gap_retrace = False
            if gap_pct is not None and gap_pct >= 2.0 and prev_close is not None:
                first_5_bars = today_bars.head(5)
                if len(first_5_bars) >= 5:
                    price_935 = float(first_5_bars["close"].iloc[-1])
                    gap_abs = day_open - prev_close  # positive (gap-up)
                    retrace_amount = day_open - price_935  # positive if price fell back
                    premarket_gap_retrace = retrace_amount > 0.5 * gap_abs

            # Intraday change (price action after the open)
            intraday_change_pct = round((current_price / day_open - 1) * 100, 2)

            # VWAP (typical price * volume method, cumulative from open)
            typical = (today_bars["high"] + today_bars["low"] + today_bars["close"]) / 3
            cum_tpv = (typical * today_bars["volume"]).cumsum()
            cum_vol = today_bars["volume"].cumsum()
            vwap = float((cum_tpv / cum_vol.replace(0, float("nan"))).iloc[-1])
            price_above_vwap = current_price > vwap
            pct_vs_vwap = round((current_price / vwap - 1) * 100, 2)

            # Intraday volume pace
            cum_volume = float(today_bars["volume"].sum())
            intraday_cumvol = int(cum_volume)

            # Opening range
            orb_end = market_open_et + timedelta(minutes=_ORB_MINUTES)
            orb_bars = today_bars[today_bars["t"] < orb_end]
            post_orb = today_bars[today_bars["t"] >= orb_end]

            orb_high = orb_low = None
            orb_breakout_up = orb_breakout_down = False

            if not orb_bars.empty:
                orb_high = round(float(orb_bars["high"].max()), 2)
                orb_low = round(float(orb_bars["low"].min()), 2)

                if not post_orb.empty and orb_high and orb_low:  # pragma: no cover
                    # Breakout: post-ORB bar closed above/below the range
                    post_closes = post_orb["close"]
                    post_vols = post_orb["volume"]
                    avg_orb_vol = float(orb_bars["volume"].mean()) if not orb_bars.empty else 1
                    orb_breakout_up = bool(
                        (post_closes > orb_high).any()
                        and float(post_vols[post_closes > orb_high].mean()) > avg_orb_vol
                    )
                    orb_breakout_down = bool((post_closes < orb_low).any())

            # Intraday RSI on 5-minute bars
            intraday_rsi = None
            five_min = today_bars.set_index("t")["close"].resample("5min").last().dropna()
            if len(five_min) >= 15:  # pragma: no cover
                import contextlib

                with contextlib.suppress(Exception):
                    intraday_rsi = round(
                        float(RSIIndicator(close=five_min, window=14).rsi().iloc[-1]), 1
                    )

            result[sym] = {
                "gap_pct": gap_pct,
                "premarket_gap_retrace": premarket_gap_retrace,
                "intraday_change_pct": intraday_change_pct,
                "intraday_cumvol": intraday_cumvol,
                "price_above_vwap": price_above_vwap,
                "pct_vs_vwap": pct_vs_vwap,
                "vwap": round(vwap, 2),
                "orb_high": orb_high,
                "orb_low": orb_low,
                "orb_breakout_up": orb_breakout_up,
                "orb_breakout_down": orb_breakout_down,
                "intraday_rsi": intraday_rsi,
            }

        except Exception as e:
            logger.debug(f"Intraday metrics failed for {sym}: {e}")
            continue

    logger.info(f"Intraday data: {len(result)}/{len(symbols)} symbols enriched")
    return result
