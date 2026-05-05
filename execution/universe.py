"""Dynamic scan universe builder.

Replaces the static config.STOCK_UNIVERSE with a daily-refreshed list drawn
from Alpaca's full asset catalog (~11,000 US equities) filtered down to
liquid, tradable symbols via the Alpaca snapshot API.

Two-stage filter:
  1. Alpaca asset catalog  → tradable + fractionable + major exchange
  2. Alpaca snapshot API   → current price ≥ $5.0 and daily volume ≥ MIN_VOLUME

Result is cached in logs/universe_cache.json with a 24-hour TTL.  On any
failure the function falls back gracefully to config.STOCK_UNIVERSE.
"""

import json
import logging
import os
from datetime import datetime

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockSnapshotRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetExchange, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, LOG_DIR, MIN_VOLUME, STOCK_UNIVERSE

logger = logging.getLogger(__name__)

_CACHE_PATH = os.path.join(LOG_DIR, "universe_cache.json")
_CACHE_TTL_HOURS = 24
_MIN_PRICE = 5.0
_MAX_UNIVERSE_SIZE = 500  # cap to keep OHLCV fetch time acceptable
_SNAPSHOT_CHUNK_SIZE = 1000  # Alpaca snapshot API limit per request

_MAJOR_EXCHANGES: frozenset[AssetExchange] = frozenset(
    {
        AssetExchange.NYSE,
        AssetExchange.NASDAQ,
        AssetExchange.ARCA,
        AssetExchange.NYSEARCA,
        AssetExchange.AMEX,
        AssetExchange.BATS,
    }
)

# Symbols confirmed delisted or unavailable from Yahoo Finance data feeds.
# Kept here to survive cache refreshes without requiring a code change.
_EXCLUDED_SYMBOLS: frozenset[str] = frozenset(
    {
        "SQ",  # Block Inc rebranded to XYZ in 2024; SQ no longer valid
        "EXAS",  # Exact Sciences: consistent yfinance data failures May 2026
    }
)


def _load_cache() -> list[str] | None:
    """Return cached symbols if the cache file exists and is within TTL."""
    try:
        with open(_CACHE_PATH) as f:
            data = json.load(f)
        saved_at = datetime.fromisoformat(data["saved_at"])
        age_h = (datetime.now() - saved_at).total_seconds() / 3600
        if age_h < _CACHE_TTL_HOURS:
            symbols = data["symbols"]
            logger.info(f"Universe cache hit: {len(symbols)} symbols (age {age_h:.1f}h)")
            return symbols
        logger.info(f"Universe cache stale ({age_h:.1f}h), refreshing")
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"Universe cache read error: {e}")
    return None


def _save_cache(symbols: list[str]) -> None:
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(_CACHE_PATH, "w") as f:
            json.dump({"saved_at": datetime.now().isoformat(), "symbols": symbols}, f)
        logger.info(f"Universe cache saved: {len(symbols)} symbols")
    except Exception as e:
        logger.warning(f"Could not save universe cache: {e}")


def _get_eligible_symbols(client: TradingClient) -> list[str]:
    """
    Return all tradable + fractionable US equity symbols on major exchanges.
    Single Alpaca API call; no market data required.
    """
    req = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
    assets = client.get_all_assets(req)
    eligible = [
        a.symbol for a in assets if a.tradable and a.fractionable and a.exchange in _MAJOR_EXCHANGES
    ]
    logger.info(f"Alpaca eligible symbols: {len(eligible)}")
    return eligible


def _apply_snapshot_filter(symbols: list[str]) -> list[str]:
    """
    Fast price + volume screen via the Alpaca snapshot API.
    One batch request per _SNAPSHOT_CHUNK_SIZE symbols; far cheaper than yfinance per-symbol calls.
    """
    data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    passed: list[str] = []

    for i in range(0, len(symbols), _SNAPSHOT_CHUNK_SIZE):
        chunk = symbols[i : i + _SNAPSHOT_CHUNK_SIZE]
        try:
            snaps = data_client.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=chunk))
            for sym, snap in snaps.items():
                bar = snap.daily_bar
                if bar is None:
                    continue
                if float(bar.close) >= _MIN_PRICE and float(bar.volume) >= MIN_VOLUME:
                    passed.append(sym)
        except Exception as e:
            logger.error(f"Snapshot filter chunk {i}–{i + _SNAPSHOT_CHUNK_SIZE} failed: {e}")
            # Fail-closed: skip unvalidated chunk rather than admitting unscreened symbols

    logger.info(f"Snapshot filter: {len(symbols)} → {len(passed)} symbols")
    return passed


def build_scan_universe(client: TradingClient) -> list[str]:
    """
    Return the daily scan universe as a deduplicated list of symbols.

    Checks the 24-hour cache first.  On a cache miss:
      1. Fetches all tradable+fractionable US equity symbols from Alpaca.
      2. Applies a price/volume screen via the Alpaca snapshot API.
      3. Merges with the core config.STOCK_UNIVERSE (always included).
      4. Caps at _MAX_UNIVERSE_SIZE symbols.

    Falls back to config.STOCK_UNIVERSE on any unexpected error.
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    try:
        eligible = _get_eligible_symbols(client)
        filtered = _apply_snapshot_filter(eligible)

        # Always include the static core; fill remaining slots from dynamic set.
        # Strip any excluded symbols (delisted / bad data) from both sources.
        core = [s for s in dict.fromkeys(STOCK_UNIVERSE) if s not in _EXCLUDED_SYMBOLS]
        core_set = set(core)
        extras = [s for s in filtered if s not in core_set and s not in _EXCLUDED_SYMBOLS]
        cap = max(0, _MAX_UNIVERSE_SIZE - len(core))
        universe = core + extras[:cap]

        logger.info(
            f"Dynamic universe built: {len(universe)} symbols "
            f"({len(core)} core + {min(len(extras), cap)} dynamic)"
        )
        _save_cache(universe)
        return universe

    except Exception as e:
        logger.error(f"Universe build failed, falling back to STOCK_UNIVERSE: {e}")
        return list(STOCK_UNIVERSE)
