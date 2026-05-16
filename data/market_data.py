import logging
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

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
from data.fundamentals import get_fundamentals

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

            ticker = yf.Ticker(symbol)
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
        # Squeeze: bandwidth in the lowest quintile of the last 20 bars
        bw_q20 = df["bb_bandwidth"].rolling(20, min_periods=5).quantile(0.2)
        df["bb_squeeze"] = df["bb_bandwidth"] <= bw_q20

        # Volume vs 20-day average
        df["avg_volume_20"] = volume.rolling(20).mean()
        df["vol_ratio"] = volume / df["avg_volume_20"]

        # Price momentum
        df["ret_1d"] = close.pct_change(1) * 100
        df["ret_5d"] = close.pct_change(5) * 100
        df["ret_10d"] = close.pct_change(10) * 100

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
        "ret_5d_pct": round(float(latest["ret_5d"]), 2),
        "ret_10d_pct": round(float(latest["ret_10d"]), 2),
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
    }


def get_vix() -> float | None:
    """Return the latest VIX close."""
    try:
        hist = yf.Ticker("^VIX").history(period="3d")
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 1)
    except Exception:
        pass
    return None


def get_spy_5d_return() -> float | None:
    """Return SPY's 5-day return % for relative strength calculation."""
    try:
        hist = yf.Ticker("SPY").history(period="15d")
        if len(hist) >= 6:
            return round(
                (float(hist["Close"].iloc[-1]) / float(hist["Close"].iloc[-6]) - 1) * 100, 2
            )
    except Exception:
        pass
    return None


def get_spy_10d_return() -> float | None:
    """Return SPY's 10-day return % for relative strength calculation."""
    try:
        hist = yf.Ticker("SPY").history(period="25d")
        if len(hist) >= 11:
            return round(
                (float(hist["Close"].iloc[-1]) / float(hist["Close"].iloc[-11]) - 1) * 100, 2
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


def _bulk_download(symbols: list[str], fetch_days: int) -> dict[str, pd.DataFrame]:
    """Download OHLCV for all symbols in a single yf.download() call.

    Using one session for all symbols avoids the per-symbol crumb/session churn
    that triggers Yahoo Finance 401 errors under concurrent load.
    """
    end = datetime.now()
    start = (end - timedelta(days=fetch_days)).strftime("%Y-%m-%d")
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
        logger.warning(f"Bulk yfinance download failed, falling back to per-symbol: {e}")
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
    return result


def get_market_snapshots(
    symbols: list[str],
    days: int = 30,
    preloaded: dict | None = None,
    as_of: str | None = None,
) -> list[dict]:
    """Fetch and summarise data for all symbols in parallel."""
    live_bulk: dict[str, pd.DataFrame] | None = None
    fundamentals: dict[str, dict] = {}
    if preloaded is not None and as_of is not None:
        spy_5d = _spy_return_from_preloaded(preloaded, as_of, 5)
        spy_10d = _spy_return_from_preloaded(preloaded, as_of, 10)
    else:
        spy_5d = get_spy_5d_return()
        spy_10d = get_spy_10d_return()
        # Single bulk download replaces 75+ parallel Ticker.history() calls
        fetch_days = max(days + 150, 200)
        live_bulk = _bulk_download(symbols, fetch_days)
        if live_bulk:
            logger.info(f"Bulk download: {len(live_bulk)}/{len(symbols)} symbols fetched")
        else:
            logger.warning("Bulk download returned no data — per-symbol fallback active")
        # Fundamentals from FMP — cached 24h, negligible cost after first fill
        fundamentals = get_fundamentals(symbols)

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
        if sym in fundamentals:
            snap.update(fundamentals[sym])
        return snap

    snapshots = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for snap in executor.map(_fetch_one, symbols):
            if snap is not None:
                snapshots.append(snap)
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
        data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
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
