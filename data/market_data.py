import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def fetch_stock_data(symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data and compute technical indicators for a symbol."""
    try:
        end = datetime.now()
        # Weekly EMA21 needs 21 weeks (~150 calendar days); fetch enough for both daily and weekly indicators
        fetch_days = max(days + 150, 200)
        start = end - timedelta(days=fetch_days)

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))

        if df.empty or len(df) < 35:
            logger.warning(f"Insufficient data for {symbol}")
            return None

        last_date = df.index[-1].date()
        staleness = (datetime.now().date() - last_date).days
        if staleness > 3:
            logger.warning(f"{symbol}: last data point {last_date} is {staleness} days old — skipping stale feed")
            return None

        df = df.copy()

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
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

        # Volume vs 20-day average
        df["avg_volume_20"] = volume.rolling(20).mean()
        df["vol_ratio"] = volume / df["avg_volume_20"]

        # Price momentum
        df["ret_1d"] = close.pct_change(1) * 100
        df["ret_5d"] = close.pct_change(5) * 100
        df["ret_10d"] = close.pct_change(10) * 100

        df = df.dropna(subset=["rsi", "macd"])

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

        # Return only the requested number of most recent days
        return df.tail(days) if len(df) >= 2 else None

    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None


def summarise_for_ai(symbol: str, df: pd.DataFrame) -> dict:
    """Extract a compact summary of the latest technical snapshot."""
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    return {
        "symbol": symbol,
        "current_price": round(float(latest["Close"]), 2),
        "ret_1d_pct": round(float(latest["ret_1d"]), 2),
        "ret_5d_pct": round(float(latest["ret_5d"]), 2),
        "ret_10d_pct": round(float(latest["ret_10d"]), 2),
        "rsi_14": round(float(latest["rsi"]), 1),
        "macd_diff": round(float(latest["macd_diff"]), 4),
        "macd_crossed_up": bool(latest["macd_diff"] > 0 and prev["macd_diff"] <= 0),
        "macd_crossed_down": bool(latest["macd_diff"] < 0 and prev["macd_diff"] >= 0),
        "ema9_above_ema21": bool(latest["ema9"] > latest["ema21"]),
        "bb_pct": round(float(latest["bb_pct"]), 2),   # 0=oversold zone, 1=overbought zone
        "vol_ratio": round(float(latest["vol_ratio"]), 2),  # >1.5 = high volume
        "avg_volume": int(float(_av)) if (_av := latest.get("avg_volume_20")) is not None and not pd.isna(_av) else 0,
        "price_vs_ema9_pct": round((float(latest["Close"]) / float(latest["ema9"]) - 1) * 100, 2),
        "weekly_trend_up": bool(latest.get("weekly_trend_up", True)),
        "weekly_rsi": float(latest.get("weekly_rsi", 50.0)),
    }


def get_vix() -> Optional[float]:
    """Return the latest VIX close."""
    try:
        hist = yf.Ticker("^VIX").history(period="3d")
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 1)
    except Exception:
        pass
    return None


def get_spy_5d_return() -> Optional[float]:
    """Return SPY's 5-day return % for relative strength calculation."""
    try:
        hist = yf.Ticker("SPY").history(period="15d")
        if len(hist) >= 6:
            return round((float(hist["Close"].iloc[-1]) / float(hist["Close"].iloc[-6]) - 1) * 100, 2)
    except Exception:
        pass
    return None


def get_market_snapshots(symbols: list[str], days: int = 30) -> list[dict]:
    """Fetch and summarise data for all symbols in parallel."""
    spy_5d = get_spy_5d_return()

    def _fetch_one(sym: str):
        df = fetch_stock_data(sym, days)
        if df is None:
            return None
        snap = summarise_for_ai(sym, df)
        if spy_5d is not None:
            snap["rel_strength_5d"] = round(snap["ret_5d_pct"] - spy_5d, 2)
        return snap

    snapshots = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        for snap in executor.map(_fetch_one, symbols):
            if snap is not None:
                snapshots.append(snap)
    return snapshots
