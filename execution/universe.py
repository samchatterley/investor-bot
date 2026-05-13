"""Dynamic scan universe builder.

Three-tier scan universe, rebuilt daily (24-hour cache):

  Tier 1 — Static core (config.STOCK_UNIVERSE)
      Curated mega-cap and high-momentum names; always included.

  Tier 2 — S&P 500 components (Wikipedia, refreshed weekly)
      Guarantees sector diversity and mid-cap growth coverage.
      Filtered to symbols tradable and fractionable on Alpaca.

  Tier 3 — Dynamic Alpaca extras
      Remaining fractionable US equities that pass a price + previous-day
      volume screen via the Alpaca snapshot API.  Fills the gap between
      the core lists and _MAX_UNIVERSE_SIZE.

Result is cached in logs/universe_cache.json (24-hour TTL).  On any
failure each tier degrades independently; the fallback is config.STOCK_UNIVERSE.
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
_SP500_CACHE_PATH = os.path.join(LOG_DIR, "sp500_cache.json")
_CACHE_TTL_HOURS = 24
_SP500_CACHE_TTL_DAYS = 7
_MIN_PRICE = 5.0
_MAX_UNIVERSE_SIZE = 750  # prefilter_candidates handles the final quality gate
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
        a.symbol  # type: ignore[union-attr]
        for a in assets
        if a.tradable and a.fractionable and a.exchange in _MAJOR_EXCHANGES  # type: ignore[union-attr]
    ]
    logger.info(f"Alpaca eligible symbols: {len(eligible)}")
    return eligible


def _fetch_sp500_symbols(eligible_set: set[str]) -> list[str]:
    """Fetch S&P 500 components from Wikipedia (7-day cache).

    Cross-references against Alpaca's fractionable set so every returned
    symbol is actually tradable on the broker.  Wikipedia uses dots in class
    share tickers (BRK.B); these are normalised to hyphens (BRK-B) to match
    the yfinance convention used throughout the codebase.
    """
    try:
        with open(_SP500_CACHE_PATH) as f:
            data = json.load(f)
        saved_at = datetime.fromisoformat(data["saved_at"])
        age_d = (datetime.now() - saved_at).total_seconds() / 86400
        if age_d < _SP500_CACHE_TTL_DAYS:
            symbols = data["symbols"]
            logger.info(f"S&P 500 cache hit: {len(symbols)} symbols (age {age_d:.1f}d)")
            return symbols
    except (FileNotFoundError, KeyError, ValueError):
        pass
    except Exception as e:
        logger.warning(f"S&P 500 cache read error: {e}")

    try:
        import pandas as pd  # lazy — only needed for Wikipedia HTML parsing

        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        raw = [s for s in tables[0]["Symbol"].tolist() if isinstance(s, str)]
        # Normalise dots → hyphens (yfinance convention); try both forms against Alpaca
        normalised = []
        for sym in raw:
            yf_sym = sym.replace(".", "-")
            dot_sym = sym  # original Wikipedia form (e.g. BRK.B)
            if yf_sym in eligible_set or dot_sym in eligible_set:
                normalised.append(yf_sym)

        logger.info(f"S&P 500: {len(raw)} from Wikipedia → {len(normalised)} Alpaca-tradable")
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(_SP500_CACHE_PATH, "w") as f:
            json.dump({"saved_at": datetime.now().isoformat(), "symbols": normalised}, f)
        return normalised

    except Exception as e:
        logger.warning(f"S&P 500 fetch failed (non-fatal): {e}")
        return []


def _apply_snapshot_filter(symbols: list[str]) -> list[str]:
    """Price + volume screen via the Alpaca snapshot API.

    Uses previous_daily_bar (yesterday's complete session) for volume.
    daily_bar reflects only the current intraday session and is near-zero
    at market open, which would cause almost every symbol to fail the
    MIN_VOLUME threshold.  Falls back to daily_bar if previous is absent.

    One batch request per _SNAPSHOT_CHUNK_SIZE symbols.
    Fail-closed: a chunk that errors is dropped rather than admitted unscreened.
    """
    data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    passed: list[str] = []

    for i in range(0, len(symbols), _SNAPSHOT_CHUNK_SIZE):
        chunk = symbols[i : i + _SNAPSHOT_CHUNK_SIZE]
        try:
            snaps = data_client.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=chunk))
            for sym, snap in snaps.items():
                bar = snap.previous_daily_bar or snap.daily_bar
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
    """Return the daily scan universe as a deduplicated list of symbols.

    Checks the 24-hour cache first.  On a cache miss, builds the three-tier
    universe (static core → S&P 500 → dynamic Alpaca extras), saves the
    result, and returns it.  Falls back to config.STOCK_UNIVERSE on error.
    """
    cached = _load_cache()
    if cached is not None:
        return cached

    try:
        eligible = _get_eligible_symbols(client)
        eligible_set = set(eligible)

        # Tier 1: static curated core (always included, highest priority)
        core = [s for s in dict.fromkeys(STOCK_UNIVERSE) if s not in _EXCLUDED_SYMBOLS]
        core_set = set(core)

        # Tier 2: S&P 500 for sector diversity and mid-cap coverage
        sp500 = _fetch_sp500_symbols(eligible_set)
        sp500_extras = [s for s in sp500 if s not in core_set and s not in _EXCLUDED_SYMBOLS]

        # Tier 3: dynamic Alpaca snapshot filter for names outside both cores
        known = core_set | set(sp500_extras)
        dynamic_candidates = [s for s in eligible if s not in known and s not in _EXCLUDED_SYMBOLS]
        dynamic_filtered = _apply_snapshot_filter(dynamic_candidates)

        # Assemble: core first, then S&P 500, then dynamic up to the cap
        extended_core = list(dict.fromkeys(core + sp500_extras))
        extras = [s for s in dynamic_filtered if s not in set(extended_core)]
        cap = max(0, _MAX_UNIVERSE_SIZE - len(extended_core))
        universe = extended_core + extras[:cap]

        logger.info(
            f"Dynamic universe built: {len(universe)} symbols "
            f"({len(core)} core + {len(sp500_extras)} S&P500 + {min(len(extras), cap)} dynamic)"
        )
        _save_cache(universe)
        return universe

    except Exception as e:
        logger.error(f"Universe build failed, falling back to STOCK_UNIVERSE: {e}")
        return list(STOCK_UNIVERSE)
