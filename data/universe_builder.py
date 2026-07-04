"""Dynamic rules-based tradeable universe — a self-maintaining replacement for the hardcoded list.

Instead of a hand-curated STOCK_UNIVERSE that goes stale (misses new listings, keeps delisted
names), this builds a liquid US-equity universe from Alpaca's free assets API + a screen. It
auto-includes new listings and drops delisted ones on each refresh.

Screen (all free, no price data needed):
  * asset_class = US_EQUITY, status = ACTIVE, tradable
  * fractionable = True         → Alpaca's own liquidity/establishment proxy (they only
                                  fractionalise liquid, established names) — a strong free filter
  * primary exchange in {NYSE, NASDAQ, AMEX}   → drop ARCA/BATS (ETF/derivative venues)
  * exclude ETFs/funds/trusts by name, and warrants/units/rights/preferreds by symbol shape

The live price band (MIN/MAX_PRICE_USD) and spread/liquidity gates still apply downstream, so this
only needs to produce the *candidate* set. Screen logic is a pure function (unit-testable); the
single Alpaca call is isolated for mocking. Cached per calendar day.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date

from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, LOG_DIR

logger = logging.getLogger(__name__)

_CACHE_PATH = os.path.join(LOG_DIR, "caching", "dynamic_universe.json")

_DEFAULT_EXCHANGES = frozenset({"NYSE", "NASDAQ", "AMEX"})
# Name markers that flag an ETF / ETN / closed-end fund / trust rather than an operating company.
_FUND_NAME_MARKERS = (
    "ETF",
    "ETN",
    " FUND",
    " TRUST",
    "PORTFOLIO",
    "ISHARES",
    "SPDR",
    "PROSHARES",
    "INVESCO",
    "VANGUARD",
    "DIREXION",
    "INDEX FUND",
)


def _is_operating_stock(symbol: str, name: str) -> bool:
    """Heuristic: keep operating-company common stock; drop ETFs/funds and warrants/units/rights."""
    if not symbol or not symbol.isalpha() or len(symbol) > 5:
        return False  # dots/hyphens/digits → preferreds, warrants, units, class shares w/ suffixes
    if symbol[-1] in ("W", "U", "R") and len(symbol) == 5:
        return False  # 5-letter ...W/U/R → warrant / unit / right
    upper = (name or "").upper()
    return not any(m in upper for m in _FUND_NAME_MARKERS)


def _screen_assets(
    assets: list, exchanges: frozenset[str], require_fractionable: bool
) -> list[str]:
    """Pure: apply the universe screen to a list of Alpaca asset objects → sorted symbol list."""
    out: set[str] = set()
    for a in assets:
        if not getattr(a, "tradable", False):
            continue
        if require_fractionable and not getattr(a, "fractionable", False):
            continue
        exch = getattr(getattr(a, "exchange", None), "value", str(getattr(a, "exchange", "")))
        if exch not in exchanges:
            continue
        sym = str(getattr(a, "symbol", ""))
        if not _is_operating_stock(sym, getattr(a, "name", "")):
            continue
        out.add(sym)
    return sorted(out)


def _fetch_assets() -> list:
    """Fetch all active US-equity assets from Alpaca (the one network call — mocked in tests)."""
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.trading.requests import GetAssetsRequest

    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    req = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
    return list(client.get_all_assets(req))


def _load_cache() -> dict:
    try:
        with open(_CACHE_PATH) as f:
            return dict(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(day: str, symbols: list[str]) -> None:
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, "w") as f:
        json.dump({"day": day, "symbols": symbols}, f)


def build_universe(
    exchanges: frozenset[str] = _DEFAULT_EXCHANGES,
    require_fractionable: bool = True,
    use_cache: bool = True,
) -> list[str]:
    """Return the dynamic liquid US-equity universe (sorted symbols). Cached per calendar day.

    Falls back to [] on Alpaca failure so callers can keep the static list as a safety net.
    """
    today = date.today().isoformat()
    if use_cache:
        cached = _load_cache()
        if cached.get("day") == today and cached.get("symbols"):
            return list(cached["symbols"])
    try:
        assets = _fetch_assets()
    except Exception as exc:
        logger.warning("universe_builder: Alpaca assets fetch failed: %s", exc)
        return []
    symbols = _screen_assets(assets, exchanges, require_fractionable)
    if use_cache and symbols:
        _save_cache(today, symbols)
    return symbols
