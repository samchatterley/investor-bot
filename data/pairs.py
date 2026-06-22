"""Cointegration-based pairs trading utilities.

Public API:
  get_cointegrated_pairs(symbols, max_age_days) -> list[dict]
  compute_zscore(sym_a, sym_b, hedge_ratio, lookback_days) -> dict | None
  refresh_pairs(symbols, lookback_days) -> list[dict]

Pairs are grouped by sector, tested with Engle-Granger cointegration, and
cached to disk at _CACHE_PATH.  All functions degrade gracefully on failure.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from itertools import combinations

import yfinance as yf

from config import LOG_DIR
from data.sector_data import SECTOR_MAP, get_sector

logger = logging.getLogger(__name__)

_CACHE_PATH = os.path.join(LOG_DIR, "caching", "pairs_cache.json")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _load_cache() -> dict:
    """Load the pairs cache from disk. Returns {} on miss or corruption."""
    try:
        with open(_CACHE_PATH) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("pairs_cache: failed to load: %s", exc)
        return {}


def _save_cache(data: dict) -> None:
    """Write pairs cache to disk. Silently swallows OSError."""
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w") as fh:
            json.dump(data, fh)
    except OSError as exc:
        logger.warning("pairs_cache: failed to save: %s", exc)


def _cache_age_days(cache: dict) -> float:
    """Return fractional age in days of the cache, or infinity if absent/corrupt."""
    generated_at = cache.get("generated_at")
    if not generated_at:
        return float("inf")
    try:
        ts = datetime.fromisoformat(generated_at)
        if ts.tzinfo is None:
            return float("inf")
        age = (datetime.now(UTC) - ts).total_seconds() / 86400
        return age
    except (ValueError, TypeError):
        return float("inf")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def get_cointegrated_pairs(
    symbols: list[str],
    max_age_days: int = 7,
) -> list[dict]:
    """Return cointegrated pairs, using a disk cache when fresh enough.

    Returns cached result if the cache is younger than max_age_days.
    Otherwise calls refresh_pairs() to rebuild and re-cache.

    Each returned dict has keys:
        sym_a, sym_b, sector, pvalue, hedge_ratio, spread_mean, spread_std
    """
    try:
        cache = _load_cache()
        if _cache_age_days(cache) < max_age_days:
            return cache.get("pairs", [])
        return refresh_pairs(symbols)
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_cointegrated_pairs failed: %s", exc)
        return []


def compute_zscore(
    sym_a: str,
    sym_b: str,
    hedge_ratio: float,
    lookback_days: int = 60,
) -> dict | None:
    """Compute current z-score of the spread between sym_a and sym_b.

    Fetches the last lookback_days of daily close prices for both symbols,
    computes spread = price_a - hedge_ratio * price_b, then returns
    {"zscore": float, "spread": float, "hedge_ratio": float}.

    Returns None on any failure (network error, too few rows, etc.).
    """
    try:
        period_days = lookback_days + 20  # buffer for weekends / holidays
        data = yf.download(
            [sym_a, sym_b],
            period=f"{period_days}d",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if data.empty:
            logger.warning("compute_zscore: no data returned for %s/%s", sym_a, sym_b)
            return None

        close = data["Close"]

        # Handle both single-level and multi-level columns
        if sym_a not in close.columns or sym_b not in close.columns:
            logger.warning("compute_zscore: symbols missing from data %s/%s", sym_a, sym_b)
            return None

        prices_a = close[sym_a].dropna()
        prices_b = close[sym_b].dropna()

        # Align on common index
        aligned = prices_a.to_frame("a").join(prices_b.to_frame("b"), how="inner").dropna()

        if len(aligned) < 2:
            logger.warning("compute_zscore: too few rows for %s/%s", sym_a, sym_b)
            return None

        spread = aligned["a"] - hedge_ratio * aligned["b"]
        mean = float(spread.mean())
        std = float(spread.std())

        if std == 0:
            logger.warning("compute_zscore: zero std for %s/%s", sym_a, sym_b)
            return None

        current_spread = float(spread.iloc[-1])
        zscore = (current_spread - mean) / std

        return {
            "zscore": zscore,
            "spread": current_spread,
            "hedge_ratio": hedge_ratio,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("compute_zscore failed for %s/%s: %s", sym_a, sym_b, exc)
        return None


def refresh_pairs(
    symbols: list[str],
    lookback_days: int = 252,
) -> list[dict]:
    """Fetch price history, run Engle-Granger within each sector, cache and return pairs.

    Steps:
    1. Resolve sector for every symbol (get_sector → SECTOR_MAP fallback; skip unknowns).
    2. Download lookback_days of daily closes for all symbols in one call.
    3. For every within-sector pair, run statsmodels coint().
    4. Keep pairs with pvalue < 0.05; fit OLS hedge ratio; compute spread stats.
    5. Write result to cache and return.

    Returns [] on any unrecoverable failure.
    """
    try:
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tsa.stattools import coint

        # --- resolve sectors ---------------------------------------------------
        sector_map: dict[str, str] = {}
        for sym in symbols:
            sector = get_sector(sym)
            if not sector or sector == "Unknown":
                sector = SECTOR_MAP.get(sym, "")
            if sector and sector != "Unknown":
                sector_map[sym] = sector
            else:
                logger.warning("refresh_pairs: skipping %s — sector unknown", sym)

        if not sector_map:
            logger.warning("refresh_pairs: no symbols with known sectors")
            return []

        known_symbols = list(sector_map.keys())

        # --- fetch price data --------------------------------------------------
        period_days = lookback_days + 20
        data = yf.download(
            known_symbols,
            period=f"{period_days}d",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )

        if data.empty:
            logger.warning("refresh_pairs: yfinance returned empty data")
            return []

        close = data["Close"]

        # --- group by sector and test all within-sector pairs ------------------
        from collections import defaultdict

        sectors: dict[str, list[str]] = defaultdict(list)
        for sym, sector in sector_map.items():
            if sym in close.columns:
                sectors[sector].append(sym)

        pairs: list[dict] = []

        for sector, syms in sectors.items():
            if len(syms) < 2:
                continue
            for sym_a, sym_b in combinations(syms, 2):
                try:
                    series_a = close[sym_a].dropna()
                    series_b = close[sym_b].dropna()

                    # Align on common index
                    aligned = (
                        series_a.to_frame("a").join(series_b.to_frame("b"), how="inner").dropna()
                    )

                    if len(aligned) < 30:
                        continue

                    price_a = aligned["a"].values
                    price_b = aligned["b"].values

                    # Engle-Granger cointegration test
                    _, pvalue, _ = coint(price_a, price_b)

                    if pvalue >= 0.05:
                        continue

                    # OLS hedge ratio: regress price_a on price_b + constant
                    import statsmodels.api as sm

                    exog = sm.add_constant(price_b)
                    result = OLS(price_a, exog).fit()
                    hedge_ratio = float(result.params[1])

                    # Spread statistics over the full lookback window
                    spread = price_a - hedge_ratio * price_b
                    spread_mean = float(spread.mean())
                    spread_std = float(spread.std())

                    pairs.append(
                        {
                            "sym_a": sym_a,
                            "sym_b": sym_b,
                            "sector": sector,
                            "pvalue": float(pvalue),
                            "hedge_ratio": hedge_ratio,
                            "spread_mean": spread_mean,
                            "spread_std": spread_std,
                        }
                    )

                except Exception as exc:  # noqa: BLE001
                    logger.warning("refresh_pairs: skipping pair %s/%s: %s", sym_a, sym_b, exc)
                    continue

        # --- persist to cache --------------------------------------------------
        cache_payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "pairs": pairs,
        }
        _save_cache(cache_payload)
        logger.info("refresh_pairs: found %d cointegrated pair(s)", len(pairs))
        return pairs

    except Exception as exc:  # noqa: BLE001
        logger.warning("refresh_pairs failed: %s", exc)
        return []
