import json
import logging
import os
import time

import numpy as np
import pandas as pd
import yfinance as yf

from config import LOG_DIR, STOCK_UNIVERSE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sector cache paths / TTL
# ---------------------------------------------------------------------------

_SECTOR_CACHE_PATH = os.path.join(LOG_DIR, "sector_map_cache.json")
_SECTOR_CACHE_TTL_DAYS = 7

# ---------------------------------------------------------------------------
# Legacy SECTOR_MAP — retained so existing callers that import it directly
# continue to work. get_sector() now uses the dynamic cache instead.
# ---------------------------------------------------------------------------

SECTOR_MAP: dict[str, str] = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "AMZN": "Consumer Discretionary",
    "META": "Technology",
    "NVDA": "Technology",
    "TSLA": "Consumer Discretionary",
    "AMD": "Technology",
    "NFLX": "Consumer Discretionary",
    "CRM": "Technology",
    "ADBE": "Technology",
    "UBER": "Industrials",
    "JPM": "Financials",
    "BAC": "Financials",
    "GS": "Financials",
    "XOM": "Energy",
    "CVX": "Energy",
    "SPY": "ETF",
    "QQQ": "ETF",
    "IWM": "ETF",
    "COST": "Consumer Staples",
    "WMT": "Consumer Staples",
    "HD": "Consumer Discretionary",
    "V": "Financials",
    "MA": "Financials",
    "PYPL": "Financials",
    "SHOP": "Technology",
    "COIN": "Financials",
    "SNAP": "Technology",
    "PINS": "Technology",
    "RBLX": "Technology",
    "DIS": "Consumer Discretionary",
    "PFE": "Healthcare",
    "MRNA": "Healthcare",
    "ABBV": "Healthcare",
    "LLY": "Healthcare",
    "BA": "Industrials",
    "CAT": "Industrials",
    "GE": "Industrials",
    "LMT": "Industrials",
    "NKE": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary",
    "TGT": "Consumer Discretionary",
    "PLTR": "Technology",
    "SNOW": "Technology",
    "NET": "Technology",
    "DDOG": "Technology",
    "CRWD": "Technology",
    "INTC": "Technology",
    "QCOM": "Technology",
    "AVGO": "Technology",
    "TXN": "Technology",
}

SECTOR_ETFS: dict[str, str] = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
    "Communication Services": "XLC",
}


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------


def _load_sector_cache() -> dict[str, str]:
    """Load {symbol: sector_name} from disk cache. Returns {} on miss/corrupt."""
    try:
        with open(_SECTOR_CACHE_PATH) as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
        return {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_sector_cache(mapping: dict[str, str]) -> None:
    """Persist mapping to disk. Silently swallows OSError."""
    try:
        with open(_SECTOR_CACHE_PATH, "w") as fh:
            json.dump(mapping, fh)
    except OSError:
        pass


def _is_sector_cache_stale(cache: dict[str, str], symbols: list[str]) -> bool:
    """True if cache is missing >10% of the requested symbols."""
    if not symbols:
        return False
    missing = sum(1 for s in symbols if s not in cache)
    return missing / len(symbols) > 0.10


# ---------------------------------------------------------------------------
# yfinance fetch
# ---------------------------------------------------------------------------


def _fetch_sector_from_yfinance(symbol: str) -> str:
    """Return yfinance.Ticker(symbol).info.get('sector', 'Unknown').

    Returns 'Unknown' on any exception.
    """
    try:
        info = yf.Ticker(symbol).info
        return info.get("sector", "Unknown") or "Unknown"
    except Exception as exc:
        logger.debug("yfinance sector fetch failed for %s: %s", symbol, exc)
        return "Unknown"


# ---------------------------------------------------------------------------
# Cache build / refresh
# ---------------------------------------------------------------------------

_SECTOR_CACHE_SAVE_EVERY = 50  # persist the partial map every N symbols during a full build


def build_sector_map(
    symbols: list[str] | None = None,
    force_refresh: bool = False,
) -> dict[str, str]:
    """Build and cache a {symbol: sector} mapping for all symbols.

    - Loads cache first; only fetches missing/stale symbols from yfinance.
    - Rate-limits requests at 0.05 s per symbol.
    - Saves updated cache to disk.
    - Returns complete mapping.
    """
    if symbols is None:
        symbols = STOCK_UNIVERSE

    cache = _load_sector_cache()

    if force_refresh or _is_sector_cache_stale(cache, symbols):
        missing = [s for s in symbols if s not in cache] if not force_refresh else symbols
        for i, sym in enumerate(missing, start=1):
            cache[sym] = _fetch_sector_from_yfinance(sym)
            time.sleep(0.05)
            # Persist incrementally so a mid-build interruption (restart/crash) keeps progress;
            # the next run then resumes from the remaining symbols instead of starting over.
            if i % _SECTOR_CACHE_SAVE_EVERY == 0:
                _save_sector_cache(cache)
        _save_sector_cache(cache)

    return cache


# ---------------------------------------------------------------------------
# Public sector lookup
# ---------------------------------------------------------------------------


def get_sector(symbol: str) -> str:
    """Return sector for symbol. Uses cache; falls back to 'Unknown' on any failure."""
    try:
        cache = _load_sector_cache()
        if symbol in cache:
            return cache[symbol]
        # Fall back to legacy hardcoded map for any symbol not yet in cache
        return SECTOR_MAP.get(symbol, "Unknown")
    except Exception as exc:
        logger.debug("get_sector failed for %s: %s", symbol, exc)
        return "Unknown"


# ---------------------------------------------------------------------------
# Momentum RS ranking
# ---------------------------------------------------------------------------


def rank_sectors_by_momentum(
    lookback_configs: list[tuple[int, float]] | None = None,
) -> list[tuple[str, str, float]]:
    """Rank sectors by composite momentum score.

    lookback_configs: list of (lookback_days, weight) pairs.
    Default: [(21, 0.3), (63, 0.4), (126, 0.3)]  # 1m, 3m, 6m

    Returns list of (etf_ticker, sector_name, composite_score) sorted best first.
    Score = weighted sum of returns over each lookback period.
    Returns [] on any data failure (callers must handle empty list).
    """
    if lookback_configs is None:
        lookback_configs = [(21, 0.3), (63, 0.4), (126, 0.3)]

    max_lookback = max(lb for lb, _ in lookback_configs)
    period_days = max_lookback + 20  # buffer for weekends / holidays

    try:
        etfs = list(SECTOR_ETFS.values())
        data = yf.download(
            etfs,
            period=f"{period_days}d",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if data.empty or len(data) < 2:
            return []

        close = data["Close"]

        scores: list[tuple[str, str, float]] = []
        for sector, etf in SECTOR_ETFS.items():
            if etf not in close.columns:
                continue
            series = close[etf].dropna()
            composite = 0.0
            valid = True
            for lookback, weight in lookback_configs:
                if len(series) <= lookback:
                    valid = False
                    break
                with np.errstate(invalid="ignore", divide="ignore"):
                    ret = series.iloc[-1] / series.iloc[-lookback] - 1
                if pd.isna(ret):
                    valid = False
                    break
                composite += weight * float(ret)
            if valid:
                scores.append((etf, sector, composite))

        scores.sort(key=lambda x: x[2], reverse=True)
        return scores

    except Exception as exc:
        logger.debug("rank_sectors_by_momentum failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Sector gate helpers
# ---------------------------------------------------------------------------


def get_sector_etf(symbol: str) -> str | None:
    """Return the SPDR ETF ticker for symbol's sector, or None if sector unknown."""
    sector = get_sector(symbol)
    return SECTOR_ETFS.get(sector)


def sector_is_leading(symbol: str, top_n: int = 4) -> bool:
    """Return True if symbol's sector is in the top_n by composite momentum.

    Fails open: returns True when sector data unavailable (don't block trades
    on data failure).
    """
    try:
        ranked = rank_sectors_by_momentum()
        if not ranked:
            return True
        sector = get_sector(symbol)
        top_sectors = {s for _, s, _ in ranked[:top_n]}
        return sector in top_sectors
    except Exception:
        return True


def sector_is_lagging(symbol: str, bottom_n: int = 3) -> bool:
    """Return True if symbol's sector is in the bottom_n by composite momentum.

    Fails open: returns False when sector data unavailable.
    """
    try:
        ranked = rank_sectors_by_momentum()
        if not ranked:
            return False
        sector = get_sector(symbol)
        bottom_sectors = {s for _, s, _ in ranked[-bottom_n:]}
        return sector in bottom_sectors
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Existing public API — unchanged
# ---------------------------------------------------------------------------


def get_sector_performance(days: int = 5) -> dict[str, float]:
    """Return {sector_name: 5d_return_pct} for major sectors, sorted best to worst."""
    try:
        etfs = list(SECTOR_ETFS.values())
        data = yf.download(
            etfs, period=f"{days + 10}d", interval="1d", progress=False, auto_adjust=True
        )
        if data.empty or len(data) < 2:
            return {}

        close = data["Close"]
        perf = {}
        for sector, etf in SECTOR_ETFS.items():
            if etf in close.columns:
                ret = (close[etf].iloc[-1] / close[etf].iloc[-days] - 1) * 100
                if not pd.isna(ret):
                    perf[sector] = round(float(ret), 2)

        return dict(sorted(perf.items(), key=lambda x: x[1], reverse=True))
    except Exception as e:
        logger.error(f"Sector performance fetch failed: {e}")
        return {}


def get_leading_sectors(top_n: int = 3) -> list[str]:
    perf = get_sector_performance()
    return list(perf.keys())[:top_n]


def check_sector_concentration(symbols: list[str], max_per_sector: int = 2) -> list[str]:
    """
    Return symbols that would breach the sector concentration cap.
    Call with current held + proposed new symbols.
    """
    sector_counts: dict[str, list[str]] = {}
    for sym in symbols:
        s = get_sector(sym)
        sector_counts.setdefault(s, []).append(sym)

    breaches = []
    for sector, syms in sector_counts.items():
        if sector == "ETF":
            continue
        if len(syms) > max_per_sector:
            # Flag the excess symbols (keep the first max_per_sector)
            breaches.extend(syms[max_per_sector:])
    return breaches
