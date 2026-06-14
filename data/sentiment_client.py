"""Market sentiment indicators from multiple free sources.

Provides:
  SentimentSnapshot  — composite sentiment reading with component breakdown
  get_aaii_sentiment()       — AAII weekly bull/bear/neutral survey
  get_fear_greed_composite() — 7-component Fear & Greed index (0–100)
  get_google_trends()        — pytrends weekly search volume for a ticker
  get_sentiment_snapshot()   — all three combined into a single SentimentSnapshot

Fear & Greed components (all from free data):
  1. SPY momentum    — SPY vs 125-day SMA (price momentum)
  2. Breadth         — % NYSE stocks above 50-day SMA (from breadth module)
  3. Put/call ratio  — SPY options PC ratio (fear hedging)
  4. VIX vs MA50     — VIX relative to its 50-day moving average
  5. Safe-haven demand — TLT vs SPY 10-day performance spread
  6. Junk bond demand — HYG/IEF ratio 10-day trend (risk appetite)
  7. NH/NL ratio     — 52-week new highs vs new lows (from breadth module)

AAII data source: https://www.aaii.com/files/surveys/sentiment.xls (public CSV).
Cache: logs/sentiment_cache.json, AAII refreshed weekly, Fear & Greed daily.
All functions degrade gracefully — return None/False on data failure.
"""

from __future__ import annotations

import io
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

from config import LOG_DIR, today_et

logger = logging.getLogger(__name__)

_CACHE_PATH = os.path.join(LOG_DIR, "sentiment_cache.json")

try:
    from pytrends.request import TrendReq as _TrendReq
except ImportError:
    _TrendReq = None
_AAII_URL = "https://www.aaii.com/files/surveys/sentiment.xls"
_AAII_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.aaii.com/sentimentsurvey",
}
_AAII_CACHE_TTL_DAYS = 7  # AAII publishes weekly
_FG_CACHE_TTL_DAYS = 1
_TRENDS_CACHE_TTL_DAYS = 7


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AAIISentiment:
    bullish_pct: float  # 0.0–1.0
    bearish_pct: float  # 0.0–1.0
    neutral_pct: float  # 0.0–1.0
    bull_bear_spread: float  # bullish_pct - bearish_pct
    extreme_bearish: bool  # bearish_pct >= 0.50 for ≥2 consecutive weeks
    extreme_bullish: bool  # bullish_pct >= 0.60 for ≥3 consecutive weeks


@dataclass(frozen=True)
class FearGreedSnapshot:
    score: float  # 0 (extreme fear) to 100 (extreme greed)
    label: str  # "Extreme Fear" | "Fear" | "Neutral" | "Greed" | "Extreme Greed"
    extreme_fear: bool  # score < 20
    extreme_greed: bool  # score > 80
    components: dict  # raw component values for diagnostics


@dataclass(frozen=True)
class SentimentSnapshot:
    aaii: AAIISentiment | None
    fear_greed: FearGreedSnapshot | None
    contrarian_long_signal: bool  # extreme_fear OR aaii extreme_bearish
    contrarian_short_signal: bool  # extreme_greed OR aaii extreme_bullish


# ── Cache helpers ─────────────────────────────────────────────────────────────


def _load_cache() -> dict:
    try:
        with open(_CACHE_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(_CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except OSError as exc:
        logger.warning("sentiment_client: cache write error: %s", exc)


def _is_stale(entry: dict, ttl_days: int) -> bool:
    try:
        fetched = entry.get("_fetched_date", "")
        age = (today_et() - datetime.strptime(fetched, "%Y-%m-%d").date()).days
        return age >= ttl_days
    except (ValueError, TypeError):
        return True


# ── AAII Sentiment Survey ─────────────────────────────────────────────────────


def _fetch_aaii_raw() -> pd.DataFrame | None:
    """Download and parse the AAII sentiment XLS. Returns DataFrame or None."""
    try:
        resp = requests.get(_AAII_URL, headers=_AAII_HEADERS, timeout=20)
        resp.raise_for_status()
        # AAII provides an XLS file with multiple tabs
        df = pd.read_excel(io.BytesIO(resp.content), sheet_name=0, header=None)
        return df
    except Exception as exc:
        logger.warning("aaii: download failed: %s", exc)
        return None


def _parse_aaii_df(df: pd.DataFrame) -> list[dict] | None:
    """Extract (date, bullish_pct, bearish_pct, neutral_pct) rows from AAII XLS.

    The AAII XLS layout has changed over the years; this parser handles the
    most common format: date in first column, bullish in second, neutral in
    third, bearish in fourth (all as decimals 0.0–1.0 or percentages).
    Returns the last 8 rows (2 months) for trend computation.
    """
    try:
        # Find the header row by looking for 'Date' or 'Bullish' keyword
        header_row = None
        for i, row in df.iterrows():
            vals = [str(v).lower().strip() for v in row.values]
            if any("bullish" in v for v in vals) and any("bearish" in v for v in vals):
                header_row = int(str(i))
                break

        if header_row is None:
            logger.debug("aaii: could not find header row")
            return None

        data = df.iloc[header_row + 1 :].copy()
        data.columns = range(len(data.columns))

        # Identify columns: first is date, then look for pct values in expected positions
        records: list[dict] = []
        for _, row in data.iterrows():
            try:
                date_val = pd.to_datetime(row[0])
                bull = float(row[1])
                neutral = float(row[2])
                bear = float(row[3])
                # Convert from percentage to decimal if needed
                if bull > 1.0:
                    bull, neutral, bear = bull / 100, neutral / 100, bear / 100
                if abs(bull + neutral + bear - 1.0) > 0.05:
                    continue
                records.append(
                    {
                        "date": date_val.date().isoformat(),
                        "bullish": round(bull, 4),
                        "neutral": round(neutral, 4),
                        "bearish": round(bear, 4),
                    }
                )
            except (ValueError, TypeError, KeyError):
                continue

        return records[-8:] if len(records) >= 2 else None

    except Exception as exc:
        logger.debug("aaii: parse failed: %s", exc)
        return None


def _build_aaii_snapshot(records: list[dict]) -> AAIISentiment:
    """Compute AAIISentiment from last N rows."""
    latest = records[-1]
    bull = float(latest["bullish"])
    bear = float(latest["bearish"])
    neutral = float(latest["neutral"])

    # Extreme bearish: ≥ 50% bears for at least the last 2 readings
    extreme_bearish = len(records) >= 2 and all(float(r["bearish"]) >= 0.50 for r in records[-2:])
    # Extreme bullish: ≥ 60% bulls for at least the last 3 readings
    extreme_bullish = len(records) >= 3 and all(float(r["bullish"]) >= 0.60 for r in records[-3:])

    return AAIISentiment(
        bullish_pct=bull,
        bearish_pct=bear,
        neutral_pct=neutral,
        bull_bear_spread=round(bull - bear, 4),
        extreme_bearish=extreme_bearish,
        extreme_bullish=extreme_bullish,
    )


def get_aaii_sentiment(force_refresh: bool = False) -> AAIISentiment | None:
    """Return the latest AAII weekly sentiment. Cached for 7 days.

    Returns None when data is unavailable.
    """
    cache = _load_cache()
    entry = cache.get("aaii", {})
    if not force_refresh and entry and not _is_stale(entry, _AAII_CACHE_TTL_DAYS):
        try:
            records = entry["records"]
            return _build_aaii_snapshot(records)
        except Exception:
            pass

    df = _fetch_aaii_raw()
    if df is None:
        return None
    records = _parse_aaii_df(df)
    if not records:
        return None

    cache["aaii"] = {"_fetched_date": today_et().isoformat(), "records": records}
    _save_cache(cache)
    logger.info(
        "aaii: latest bull=%.0f%% bear=%.0f%%",
        records[-1]["bullish"] * 100,
        records[-1]["bearish"] * 100,
    )
    return _build_aaii_snapshot(records)


# ── Fear & Greed Composite ────────────────────────────────────────────────────


def _fg_spy_momentum(spy_closes: pd.Series) -> float | None:
    """Component 1: SPY price vs 125-day SMA. Above = greed (100), below = fear (0)."""
    if len(spy_closes) < 126:
        return None
    sma125 = float(spy_closes.rolling(125).mean().iloc[-1])
    latest = float(spy_closes.iloc[-1])
    if sma125 <= 0:
        return None
    # Normalize: how far above/below the SMA, scaled to 0–100
    pct_diff = (latest / sma125 - 1) * 100
    score = min(100.0, max(0.0, 50 + pct_diff * 4))
    return round(score, 1)


def _fg_vix_vs_ma(vix_closes: pd.Series) -> float | None:
    """Component 4: VIX vs 50-day MA. VIX above MA = fear (0), below = greed (100)."""
    if len(vix_closes) < 51:
        return None
    ma50 = float(vix_closes.rolling(50).mean().iloc[-1])
    vix = float(vix_closes.iloc[-1])
    if ma50 <= 0:
        return None
    ratio = vix / ma50
    # ratio > 1 = elevated fear → score near 0; ratio < 1 = calm → score near 100
    score = min(100.0, max(0.0, (2.0 - ratio) * 50))
    return round(score, 1)


def _fg_tlt_spy_spread(tlt_closes: pd.Series, spy_closes: pd.Series) -> float | None:
    """Component 5: TLT vs SPY 10-day spread. Positive (flight to safety) = fear."""
    if len(tlt_closes) < 11 or len(spy_closes) < 11:
        return None
    tlt_10d = float((tlt_closes.iloc[-1] / tlt_closes.iloc[-11] - 1) * 100)
    spy_10d = float((spy_closes.iloc[-1] / spy_closes.iloc[-11] - 1) * 100)
    spread = tlt_10d - spy_10d
    # Spread > 0 = TLT outperforming = flight to safety = fear
    score = min(100.0, max(0.0, 50 - spread * 8))
    return round(score, 1)


def _fg_hyg_ief_trend(hyg_closes: pd.Series | None, ief_closes: pd.Series | None) -> float | None:
    """Component 6: HYG/IEF ratio 10-day ROC. Rising = greed (risk appetite)."""
    if hyg_closes is None or ief_closes is None:
        return None
    if len(hyg_closes) < 11 or len(ief_closes) < 11:
        return None
    try:
        combined = pd.concat([hyg_closes.rename("hyg"), ief_closes.rename("ief")], axis=1).dropna()
        if len(combined) < 11:
            return None
        ratio = combined["hyg"] / combined["ief"]
        roc_10d = float((ratio.iloc[-1] / ratio.iloc[-11] - 1) * 100)
        score = min(100.0, max(0.0, 50 + roc_10d * 15))
        return round(score, 1)
    except Exception:
        return None


def _download_fg_prices() -> dict[str, pd.Series]:
    """Download SPY, VIX, TLT, HYG, IEF for Fear & Greed computation."""
    tickers = ["SPY", "^VIX", "TLT", "HYG", "IEF"]
    end = datetime.now()
    start = (end - timedelta(days=220)).strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    result: dict[str, pd.Series] = {}
    try:
        raw = yf.download(
            tickers=tickers,
            start=start,
            end=end_str,
            auto_adjust=True,
            progress=False,
        )
        if raw is None or raw.empty:
            return {}
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
            for t in tickers:
                if t in close.columns:
                    s = close[t].dropna()
                    if len(s) >= 11:
                        result[t] = s
    except Exception as exc:
        logger.warning("fear_greed: download failed: %s", exc)
    return result


def compute_fear_greed(
    prices: dict[str, pd.Series],
    nh_nl_ratio: float | None = None,
    pct_above_sma50: float | None = None,
) -> FearGreedSnapshot:
    """Compute 7-component Fear & Greed composite from price series dict.

    Components 2 (breadth) and 3 (put/call) are provided as optional
    scalar inputs from the breadth module / options scanner.  When absent,
    those components are omitted from the composite.

    Args:
        prices: {ticker: close_series} from _download_fg_prices()
        nh_nl_ratio: new_highs / (new_lows + 1) from breadth module
        pct_above_sma50: fraction of universe stocks above 50d SMA
    """
    spy = prices.get("SPY")
    vix = prices.get("^VIX")
    tlt = prices.get("TLT")
    hyg = prices.get("HYG")
    ief = prices.get("IEF")

    components: dict[str, float | None] = {}

    # Component 1: SPY momentum vs 125-day SMA
    components["spy_momentum"] = _fg_spy_momentum(spy) if spy is not None else None

    # Component 2: breadth (% above SMA50) → 0–100 score
    if pct_above_sma50 is not None:
        components["breadth"] = round(min(100.0, pct_above_sma50 * 100), 1)
    else:
        components["breadth"] = None

    # Component 3: NH/NL ratio → 0–100 score (nh_nl_ratio > 2 = greed)
    if nh_nl_ratio is not None:
        components["nh_nl"] = round(min(100.0, max(0.0, (nh_nl_ratio / 4.0) * 100)), 1)
    else:
        components["nh_nl"] = None

    # Component 4: VIX vs MA50
    components["vix_vs_ma"] = _fg_vix_vs_ma(vix) if vix is not None else None

    # Component 5: TLT vs SPY safe-haven demand
    if tlt is not None and spy is not None:
        components["safe_haven"] = _fg_tlt_spy_spread(tlt, spy)
    else:
        components["safe_haven"] = None

    # Component 6: HYG/IEF junk bond demand
    if hyg is not None and ief is not None:
        components["junk_demand"] = _fg_hyg_ief_trend(hyg, ief)
    else:
        components["junk_demand"] = None

    # Composite: equal-weight average of available components
    valid = [v for v in components.values() if v is not None]
    if not valid:
        return FearGreedSnapshot(
            score=50.0,
            label="Neutral",
            extreme_fear=False,
            extreme_greed=False,
            components=components,
        )

    score = round(sum(valid) / len(valid), 1)

    if score < 20:
        label = "Extreme Fear"
    elif score < 40:
        label = "Fear"
    elif score < 60:
        label = "Neutral"
    elif score < 80:
        label = "Greed"
    else:
        label = "Extreme Greed"

    return FearGreedSnapshot(
        score=score,
        label=label,
        extreme_fear=score < 20,
        extreme_greed=score > 80,
        components=components,
    )


def get_fear_greed_composite(
    force_refresh: bool = False,
    nh_nl_ratio: float | None = None,
    pct_above_sma50: float | None = None,
) -> FearGreedSnapshot | None:
    """Return Fear & Greed composite. Cached for 1 day.

    Returns None when no price data is available.
    """
    cache = _load_cache()
    entry = cache.get("fear_greed", {})
    if not force_refresh and entry and not _is_stale(entry, _FG_CACHE_TTL_DAYS):
        try:
            snap_data = {k: v for k, v in entry.items() if k != "_fetched_date"}
            return FearGreedSnapshot(**snap_data)
        except Exception:
            pass

    prices = _download_fg_prices()
    snapshot = compute_fear_greed(prices, nh_nl_ratio=nh_nl_ratio, pct_above_sma50=pct_above_sma50)

    cache_payload = {**asdict(snapshot), "_fetched_date": today_et().isoformat()}
    cache["fear_greed"] = cache_payload
    _save_cache(cache)
    logger.info("fear_greed: score=%.1f label=%s", snapshot.score, snapshot.label)
    return snapshot


# ── Google Trends ─────────────────────────────────────────────────────────────


def get_google_trends(symbol: str, force_refresh: bool = False) -> dict | None:
    """Fetch Google Trends weekly interest for a ticker symbol.

    Returns:
        {
            "current_interest": int,     # 0–100, last week
            "avg_interest_12w": float,   # 12-week average
            "spike": bool,               # current > avg * 2.5
            "declining": bool,           # current < avg * 0.5
        }
    Returns None when pytrends is unavailable or data missing.
    """
    cache = _load_cache()
    trends_key = f"trends_{symbol}"
    entry = cache.get(trends_key, {})
    if not force_refresh and entry and not _is_stale(entry, _TRENDS_CACHE_TTL_DAYS):
        try:
            payload = {k: v for k, v in entry.items() if k != "_fetched_date"}
            return payload
        except Exception:
            pass

    try:
        if _TrendReq is None:
            raise ImportError("pytrends not installed")

        pytrends = _TrendReq(hl="en-US", tz=300, timeout=(10, 30))
        pytrends.build_payload([symbol], cat=0, timeframe="today 3-m", geo="US")
        df = pytrends.interest_over_time()

        if df is None or df.empty or symbol not in df.columns:
            return None

        series = df[symbol].dropna()
        if len(series) < 2:
            return None

        current = int(series.iloc[-1])
        avg_12w = float(series.tail(12).mean())

        result = {
            "current_interest": current,
            "avg_interest_12w": round(avg_12w, 1),
            "spike": avg_12w > 0 and current > avg_12w * 2.5,
            "declining": avg_12w > 0 and current < avg_12w * 0.5,
        }
        cache[trends_key] = {**result, "_fetched_date": today_et().isoformat()}
        _save_cache(cache)
        return result

    except Exception as exc:
        logger.debug("google_trends: %s: %s", symbol, exc)
        return None


# ── Combined snapshot ─────────────────────────────────────────────────────────


def get_sentiment_snapshot(
    nh_nl_ratio: float | None = None,
    pct_above_sma50: float | None = None,
) -> SentimentSnapshot:
    """Fetch all sentiment indicators and return a combined SentimentSnapshot.

    Breadth inputs (nh_nl_ratio, pct_above_sma50) are optional — pass them
    from the breadth module to improve Fear & Greed accuracy.
    """
    aaii = get_aaii_sentiment()
    fg = get_fear_greed_composite(nh_nl_ratio=nh_nl_ratio, pct_above_sma50=pct_above_sma50)

    extreme_fear = (fg is not None and fg.extreme_fear) or (
        aaii is not None and aaii.extreme_bearish
    )
    extreme_greed = (fg is not None and fg.extreme_greed) or (
        aaii is not None and aaii.extreme_bullish
    )

    return SentimentSnapshot(
        aaii=aaii,
        fear_greed=fg,
        contrarian_long_signal=extreme_fear,
        contrarian_short_signal=extreme_greed,
    )
