import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

import requests

from config import FMP_API_KEY, LOG_DIR

logger = logging.getLogger(__name__)

_BASE = "https://financialmodelingprep.com/api/v3"
_FUND_CACHE = os.path.join(LOG_DIR, "fmp_fundamentals_cache.json")
_ANALYST_CACHE = os.path.join(LOG_DIR, "fmp_analyst_cache.json")
_CACHE_TTL_HOURS = 24
_MAX_WORKERS = 5


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
    if not FMP_API_KEY:
        return None
    try:
        resp = requests.get(
            f"{_BASE}{path}",
            params={"apikey": FMP_API_KEY, **(params or {})},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.debug(f"FMP {path}: {e}")
        return None


def _fetch_ratios(sym: str) -> tuple[str, dict]:
    raw = _get(f"/ratios-ttm/{sym}")
    if not raw:
        return sym, {}
    rec = raw[0] if isinstance(raw, list) and raw else (raw if isinstance(raw, dict) else {})
    return sym, {
        "roe": rec.get("returnOnEquityTTM"),
        "profit_margin": rec.get("netProfitMarginTTM"),
        "debt_to_equity": rec.get("debtEquityRatioTTM"),
        "current_ratio": rec.get("currentRatioTTM"),
    }


def _fetch_analyst(sym: str) -> tuple[str, dict]:
    recs_raw = _get(f"/analyst-stock-recommendations/{sym}", {"limit": "1"})
    targets_raw = _get(f"/price-target-consensus/{sym}")

    data: dict = {}

    if recs_raw:
        r = (
            recs_raw[0]
            if isinstance(recs_raw, list) and recs_raw
            else (recs_raw if isinstance(recs_raw, dict) else {})
        )
        # FMP uses inconsistent casing: "analystRatingsbuy" (lowercase b)
        strong_buy = int(r.get("analystRatingsStrongBuy") or 0)
        buy = int(r.get("analystRatingsbuy") or 0)
        hold = int(r.get("analystRatingsHold") or 0)
        sell = int(r.get("analystRatingsSell") or 0)
        strong_sell = int(r.get("analystRatingsStrongSell") or 0)
        total = strong_buy + buy + hold + sell + strong_sell
        if total:
            data["bullish_pct"] = round((strong_buy + buy) / total * 100)
            data["bearish_pct"] = round((sell + strong_sell) / total * 100)
            data["analyst_count"] = total

    if targets_raw:
        t = (
            targets_raw[0]
            if isinstance(targets_raw, list) and targets_raw
            else (targets_raw if isinstance(targets_raw, dict) else {})
        )
        target = t.get("targetConsensus") or t.get("targetMedian")
        if target:
            import contextlib

            with contextlib.suppress(ValueError, TypeError):
                data["target_price"] = round(float(target), 2)

    return sym, data


def get_fundamentals(symbols: list[str]) -> dict[str, dict]:
    """Fetch key financial ratios from FMP with 24-hour cache.

    Returns {symbol: {roe, profit_margin, debt_to_equity, current_ratio}}.
    Individual fields are None when FMP doesn't report them. Symbols with no
    data are omitted. Returns {} immediately when FMP_API_KEY is unset.
    """
    if not FMP_API_KEY or not symbols:
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
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            futures = {executor.submit(_fetch_ratios, sym): sym for sym in to_fetch}
            for fut in as_completed(futures):
                try:
                    sym, data = fut.result()
                    cache[sym] = {"fetched_at": now_iso, "data": data}
                    if data:
                        result[sym] = data
                except Exception as e:
                    logger.debug(f"Fundamentals fetch error: {e}")
        _save_cache(_FUND_CACHE, cache)
        logger.info(f"FMP fundamentals fetched for {len(to_fetch)} symbol(s)")

    logger.info(f"FMP fundamentals: {len(result)}/{len(symbols)} symbols")
    return result


def get_analyst_consensus(symbols: list[str]) -> dict[str, dict]:
    """Fetch analyst ratings and price targets from FMP with 24-hour cache.

    Returns {symbol: {bullish_pct, bearish_pct, analyst_count, target_price}}.
    target_price (consensus price target) is optional and may be absent.
    Returns {} immediately when FMP_API_KEY is unset.
    """
    if not FMP_API_KEY or not symbols:
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
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            futures = {executor.submit(_fetch_analyst, sym): sym for sym in to_fetch}
            for fut in as_completed(futures):
                try:
                    sym, data = fut.result()
                    cache[sym] = {"fetched_at": now_iso, "data": data}
                    if data:
                        result[sym] = data
                except Exception as e:
                    logger.debug(f"Analyst fetch error: {e}")
        _save_cache(_ANALYST_CACHE, cache)
        logger.info(f"FMP analyst consensus fetched for {len(to_fetch)} symbol(s)")

    logger.info(f"FMP analyst consensus: {len(result)}/{len(symbols)} symbols")
    return result
