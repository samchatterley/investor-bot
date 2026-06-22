"""Analyst revision signals from yfinance recommendations_summary.

Detects consensus shifts from Hold→Buy (upgrade signal) or Buy→Hold/Underperform
(downgrade signal) by comparing the current week's recommendations distribution
to the prior period.

Cache lives at logs/analyst_revisions_cache.json and is refreshed once per
calendar day. Call prefetch_analyst_revisions() from the pre-market prefetch job.

Result schema per symbol::

    {
        "analyst_upgrade":   bool,  # Hold→Buy shift in past 1 month
        "analyst_downgrade": bool,  # Buy→Hold/Sell shift in past 1 month
    }
"""

import json
import logging
import os
import time

import yfinance as yf

from config import ETF_SYMBOLS, LOG_DIR, STOCK_UNIVERSE, today_et

logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

_CACHE_PATH = os.path.join(LOG_DIR, "caching", "analyst_revisions_cache.json")
_REQ_DELAY = 0.05


# ── cache I/O ─────────────────────────────────────────────────────────────────


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
    except OSError as e:
        logger.warning(f"analyst_revisions_cache: write error: {e}")


# ── live fetch ────────────────────────────────────────────────────────────────


def _live_fetch_revisions(symbols: list[str]) -> dict[str, dict | None]:
    """Fetch analyst revision signals from yfinance for each symbol."""
    result: dict[str, dict | None] = {}

    for sym in symbols:
        if sym in ETF_SYMBOLS:
            result[sym] = None
            continue
        try:
            time.sleep(_REQ_DELAY)
            rec = yf.Ticker(sym).recommendations_summary
        except Exception as exc:
            logger.debug(f"analyst_revisions fetch failed for {sym}: {exc}")
            result[sym] = None
            continue

        if rec is None or rec.empty:
            result[sym] = None
            continue

        try:
            # recommendations_summary has period as index (e.g. "0m", "1m", "2m", "3m")
            # 0m = current month, 1m = last month
            if len(rec) < 2:
                result[sym] = None
                continue

            rec_dict = rec.to_dict(orient="index")
            periods = sorted(rec_dict.keys())  # "0m" < "1m" < "2m"
            cur = rec_dict[periods[0]]
            prev = rec_dict[periods[1]]

            cur_buy = int(cur.get("strongBuy", 0)) + int(cur.get("buy", 0))
            cur_hold = int(cur.get("hold", 0))
            cur_sell = int(cur.get("sell", 0)) + int(cur.get("strongSell", 0))
            cur_total = cur_buy + cur_hold + cur_sell

            prev_buy = int(prev.get("strongBuy", 0)) + int(prev.get("buy", 0))
            prev_sell = int(prev.get("sell", 0)) + int(prev.get("strongSell", 0))
            prev_total = prev_buy + int(prev.get("hold", 0)) + prev_sell

            if cur_total < 3 or prev_total < 3:
                result[sym] = None
                continue

            cur_buy_pct = cur_buy / cur_total
            prev_buy_pct = prev_buy / prev_total
            cur_sell_pct = cur_sell / cur_total
            prev_sell_pct = prev_sell / prev_total

            # Upgrade: buy % rose by >10pp relative to prior month
            analyst_upgrade = (cur_buy_pct - prev_buy_pct) > 0.10
            # Downgrade: sell % rose by >10pp OR buy % fell by >10pp
            analyst_downgrade = (cur_sell_pct - prev_sell_pct) > 0.10 or (
                prev_buy_pct - cur_buy_pct
            ) > 0.10

            result[sym] = {
                "analyst_upgrade": analyst_upgrade,
                "analyst_downgrade": analyst_downgrade,
            }
        except Exception as exc:
            logger.debug(f"analyst_revisions parse failed for {sym}: {exc}")
            result[sym] = None

    return result


# ── Public prefetch ────────────────────────────────────────────────────────────


def prefetch_analyst_revisions(symbols: list[str] | None = None) -> int:
    """Warm the same-day analyst revisions cache before market open."""
    if symbols is None:
        symbols = list(STOCK_UNIVERSE)
    today = today_et().isoformat()
    cache = _load_cache()
    today_cache: dict[str, dict | None] = cache.get(today, {})

    missing = [s for s in symbols if s not in today_cache]
    if not missing:
        logger.info(f"analyst_revisions prefetch: cache already warm ({len(today_cache)} symbols)")
        return 0

    fresh = _live_fetch_revisions(missing)
    today_cache.update(fresh)
    _save_cache({today: today_cache})

    found = sum(1 for v in today_cache.values() if v is not None)
    logger.info(f"analyst_revisions prefetch: fetched {len(missing)} symbols, {found} with signals")
    return len(missing)


# ── Public getter ──────────────────────────────────────────────────────────────


def get_analyst_revisions(symbols: list[str]) -> dict[str, dict]:
    """Return analyst revision data for symbols with detected upgrades/downgrades.

    Uses the same-day cache. Falls back to live fetch on cache miss.
    Symbols with no qualifying data are omitted from the result.
    """
    today = today_et().isoformat()
    cache = _load_cache()
    today_cache: dict[str, dict | None] = cache.get(today, {})

    missing = [s for s in symbols if s not in today_cache]
    if missing:
        fresh = _live_fetch_revisions(missing)
        today_cache.update(fresh)
        _save_cache({today: today_cache})

    return {sym: v for sym in symbols if (v := today_cache.get(sym)) is not None}
