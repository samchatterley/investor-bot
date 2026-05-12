from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime

import requests

from config import FINNHUB_API_KEY, LOG_DIR

logger = logging.getLogger(__name__)

_BASE = "https://finnhub.io/api/v1"
_FUND_CACHE = os.path.join(LOG_DIR, "fmp_fundamentals_cache.json")
_ANALYST_CACHE = os.path.join(LOG_DIR, "fmp_analyst_cache.json")
_CACHE_TTL_HOURS = 24


def _load_cache(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(path: str, cache: dict) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def _is_stale(entry: dict) -> bool:
    try:
        age_s = (datetime.now(UTC) - datetime.fromisoformat(entry["fetched_at"])).total_seconds()
        return age_s > _CACHE_TTL_HOURS * 3600
    except (KeyError, ValueError):
        return True


def _get(path: str, params: dict | None = None) -> dict | list | None:
    if not FINNHUB_API_KEY:
        return None
    try:
        resp = requests.get(
            f"{_BASE}{path}",
            params={"token": FINNHUB_API_KEY, **(params or {})},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.debug(f"Finnhub {path}: {e}")
        return None


def _fetch_ratios(sym: str) -> tuple[str, dict]:
    raw = _get("/stock/metric", {"symbol": sym, "metric": "all"})
    if not raw:
        return sym, {}
    m = raw.get("metric") or {}
    # Finnhub reports ROE and margins as percentages — convert to decimal ratios
    roe = m.get("roeTTM")
    margin = m.get("netProfitMarginTTM")
    data = {
        "roe": round(roe / 100, 6) if roe is not None else None,
        "profit_margin": round(margin / 100, 6) if margin is not None else None,
        "debt_to_equity": m.get("totalDebt/totalEquityAnnual"),
        "current_ratio": m.get("currentRatioAnnual"),
    }
    if all(v is None for v in data.values()):
        return sym, {}
    return sym, data


def _fetch_analyst(sym: str) -> tuple[str, dict]:
    raw = _get("/stock/recommendation", {"symbol": sym})
    if not raw or not isinstance(raw, list) or not raw:
        return sym, {}
    # Take the most recent period (first element)
    r = raw[0]
    strong_buy = int(r.get("strongBuy") or 0)
    buy = int(r.get("buy") or 0)
    hold = int(r.get("hold") or 0)
    sell = int(r.get("sell") or 0)
    strong_sell = int(r.get("strongSell") or 0)
    total = strong_buy + buy + hold + sell + strong_sell
    if not total:
        return sym, {}
    return sym, {
        "bullish_pct": round((strong_buy + buy) / total * 100),
        "bearish_pct": round((sell + strong_sell) / total * 100),
        "analyst_count": total,
    }


def get_fundamentals(symbols: list[str]) -> dict[str, dict]:
    """Fetch key financial ratios from Finnhub with 24-hour cache.

    Returns {symbol: {roe, profit_margin, debt_to_equity, current_ratio}}.
    Individual fields are None when Finnhub doesn't report them. Symbols
    with no data are omitted. Returns {} immediately when FINNHUB_API_KEY is unset.
    """
    if not FINNHUB_API_KEY or not symbols:
        return {}

    cache = _load_cache(_FUND_CACHE)
    now_iso = datetime.now(UTC).isoformat()
    result: dict[str, dict] = {}
    to_fetch: list[str] = []

    for sym in symbols:
        entry = cache.get(sym, {})
        if entry and not _is_stale(entry) and entry.get("data"):
            result[sym] = entry["data"]
        else:
            to_fetch.append(sym)

    if to_fetch:
        for sym in to_fetch:
            try:
                sym, data = _fetch_ratios(sym)
                cache[sym] = {"fetched_at": now_iso, "data": data}
                if data:
                    result[sym] = data
            except Exception as e:
                logger.debug(f"Fundamentals fetch error: {e}")
        _save_cache(_FUND_CACHE, cache)
        logger.info(f"Fundamentals fetched for {len(to_fetch)} symbol(s)")

    logger.info(f"Fundamentals: {len(result)}/{len(symbols)} symbols")
    return result


def get_analyst_consensus(symbols: list[str]) -> dict[str, dict]:
    """Fetch analyst ratings from Finnhub with 24-hour cache.

    Returns {symbol: {bullish_pct, bearish_pct, analyst_count}}.
    bullish_pct/bearish_pct are computed from exact strongBuy/buy/sell/strongSell counts.
    Returns {} immediately when FINNHUB_API_KEY is unset.
    """
    if not FINNHUB_API_KEY or not symbols:
        return {}

    cache = _load_cache(_ANALYST_CACHE)
    now_iso = datetime.now(UTC).isoformat()
    result: dict[str, dict] = {}
    to_fetch: list[str] = []

    for sym in symbols:
        entry = cache.get(sym, {})
        if entry and not _is_stale(entry) and entry.get("data"):
            result[sym] = entry["data"]
        else:
            to_fetch.append(sym)

    if to_fetch:
        for sym in to_fetch:
            try:
                sym, data = _fetch_analyst(sym)
                cache[sym] = {"fetched_at": now_iso, "data": data}
                if data:
                    result[sym] = data
            except Exception as e:
                logger.debug(f"Analyst fetch error: {e}")
        _save_cache(_ANALYST_CACHE, cache)
        logger.info(f"Analyst consensus fetched for {len(to_fetch)} symbol(s)")

    logger.info(f"Analyst consensus: {len(result)}/{len(symbols)} symbols")
    return result
