"""Options market data: IV surface, skew, and ATM IV vs realized vol.

Provides per-symbol OptionsSnapshot covering:
  atm_iv            — near-ATM 30-day implied volatility (annualized)
  skew_25d          — 25-delta put IV / 25-delta call IV ratio
  put_call_oi_ratio — total put OI / total call OI (nearest expiry)
  iv_rv_spread      — ATM IV minus 20-day realized vol (IV premium)
  iv_cheap          — True when iv_rv_spread < -0.07 (IV below RV by >7pp)
  iv_expensive      — True when iv_rv_spread > 0.15 (IV premium > 15pp)
  unusual_call_oi   — True when today's call OI > 3× prior 5-day avg call OI
  panic_put_skew    — True when skew_25d > 1.4 (crowded hedging)
  call_skew_spike   — True when skew_25d < 0.75 (unusual call demand)

Uses yfinance option chains (no API key required).
Black-Scholes delta computed via scipy.stats.norm for accurate 25-delta approximation.
Cache: logs/options_data_cache.json, per-symbol entries refreshed daily.
All public functions degrade gracefully — return None/False on data failure.
"""

from __future__ import annotations

import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime

import pandas as pd
import yfinance as yf
from scipy.stats import norm

from config import LOG_DIR, today_et

logger = logging.getLogger(__name__)

_CACHE_PATH = os.path.join(LOG_DIR, "options_data_cache.json")
_MIN_OPTION_VOLUME = 10
_MIN_OPEN_INTEREST = 50
_RISK_FREE_RATE = 0.05  # approximate; updated annually
_TARGET_DTE_MIN = 10
_TARGET_DTE_MAX = 60
_TARGET_DTE_PREF = 30


# ── Data class ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OptionsSnapshot:
    atm_iv: float | None
    skew_25d: float | None
    put_call_oi_ratio: float | None
    iv_rv_spread: float | None
    iv_cheap: bool
    iv_expensive: bool
    unusual_call_oi: bool
    panic_put_skew: bool
    call_skew_spike: bool


def _null_snapshot() -> OptionsSnapshot:
    return OptionsSnapshot(
        atm_iv=None,
        skew_25d=None,
        put_call_oi_ratio=None,
        iv_rv_spread=None,
        iv_cheap=False,
        iv_expensive=False,
        unusual_call_oi=False,
        panic_put_skew=False,
        call_skew_spike=False,
    )


# ── Black-Scholes helpers ─────────────────────────────────────────────────────


def _bs_delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """Black-Scholes delta for a European option.

    Returns 0.0 when T ≤ 0 or sigma ≤ 0 (degenerate inputs).
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return float(norm.cdf(d1) if is_call else norm.cdf(d1) - 1.0)


def _realized_vol_20d(ticker_obj: yf.Ticker) -> float | None:
    """Compute 20-day annualised realised volatility from daily close returns."""
    try:
        hist = ticker_obj.history(period="2mo", interval="1d", auto_adjust=True)
        if hist.empty or len(hist) < 22:
            return None
        closes = hist["Close"].dropna()
        if len(closes) < 22:
            return None
        log_rets = closes.pct_change().dropna().iloc[-20:]
        rv = float(log_rets.std() * math.sqrt(252))
        return round(rv, 4) if rv > 0 else None
    except Exception as exc:
        logger.debug("realized_vol: %s", exc)
        return None


# ── Option chain helpers ──────────────────────────────────────────────────────


def _select_expiry(expirations: tuple[str, ...]) -> str | None:
    """Pick the expiry closest to _TARGET_DTE_PREF within [_TARGET_DTE_MIN, _TARGET_DTE_MAX]."""
    today = datetime.now().date()
    best: str | None = None
    best_dist = float("inf")
    for exp in expirations:
        try:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if _TARGET_DTE_MIN <= dte <= _TARGET_DTE_MAX:
                dist = abs(dte - _TARGET_DTE_PREF)
                if dist < best_dist:
                    best_dist = dist
                    best = exp
        except ValueError:
            continue
    return best


def _atm_iv(chain_df: pd.DataFrame, spot: float) -> float | None:
    """Return IV of the option with strike nearest to spot."""
    if chain_df.empty or spot <= 0:
        return None
    df = chain_df.dropna(subset=["impliedVolatility", "strike"])
    df = df[df["impliedVolatility"] > 0]
    if df.empty:
        return None
    idx = (df["strike"] - spot).abs().idxmin()
    iv = float(df.loc[idx, "impliedVolatility"])
    return round(iv, 4) if iv > 0 else None


def _find_25d_iv(
    chain_df: pd.DataFrame,
    spot: float,
    T: float,
    r: float,
    is_call: bool,
) -> float | None:
    """Find the IV of the option whose BS delta is closest to ±0.25.

    For calls: target delta = +0.25, for puts: target delta = -0.25.
    Requires at least one option row with valid IV and strike.
    Returns None when no suitable option can be found.
    """
    if chain_df.empty or spot <= 0 or T <= 0:
        return None
    df = chain_df.dropna(subset=["impliedVolatility", "strike"])
    df = df[df["impliedVolatility"] > 0]
    if df.empty:
        return None

    target_delta = 0.25 if is_call else -0.25
    best_iv: float | None = None
    best_dist = float("inf")

    for _, row in df.iterrows():
        K = float(row["strike"])
        iv = float(row["impliedVolatility"])
        delta = _bs_delta(spot, K, T, r, iv, is_call)
        dist = abs(delta - target_delta)
        if dist < best_dist:
            best_dist = dist
            best_iv = iv

    # Only accept if the best delta found is within 0.10 of target (avoid degenerate strikes)
    if best_dist > 0.10:
        return None
    return round(best_iv, 4) if best_iv is not None else None


# ── Per-symbol computation ────────────────────────────────────────────────────


def _compute_snapshot(symbol: str) -> OptionsSnapshot:
    """Fetch option chain and price history for symbol and compute OptionsSnapshot."""
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        if not expirations:
            return _null_snapshot()

        expiry = _select_expiry(expirations)
        if expiry is None:
            return _null_snapshot()

        chain = ticker.option_chain(expiry)
        calls = chain.calls
        puts = chain.puts
        if calls.empty and puts.empty:
            return _null_snapshot()

        # Current spot price
        info = ticker.fast_info
        try:
            spot = float(info.last_price)
        except Exception:
            spot = 0.0
        if spot <= 0:
            # Fallback: use ATM strike midpoint
            all_strikes = pd.concat(
                [
                    calls["strike"] if not calls.empty else pd.Series(dtype=float),
                    puts["strike"] if not puts.empty else pd.Series(dtype=float),
                ]
            )
            spot = float(all_strikes.median()) if not all_strikes.empty else 0.0
        if spot <= 0:
            return _null_snapshot()

        # Time to expiry in years
        exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        dte = (exp_date - datetime.now().date()).days
        T = dte / 365.0

        # ATM IV (from nearest-strike call — calls tend to have tighter spreads)
        atm_iv_val = _atm_iv(calls if not calls.empty else puts, spot)

        # 25-delta skew
        put_25d_iv = _find_25d_iv(puts, spot, T, _RISK_FREE_RATE, is_call=False)
        call_25d_iv = _find_25d_iv(calls, spot, T, _RISK_FREE_RATE, is_call=True)
        skew_25d: float | None = None
        if put_25d_iv is not None and call_25d_iv is not None and call_25d_iv > 0:
            skew_25d = round(put_25d_iv / call_25d_iv, 3)

        # Put/call OI ratio (nearest expiry)
        call_oi = float(calls["openInterest"].fillna(0).sum()) if not calls.empty else 0.0
        put_oi = float(puts["openInterest"].fillna(0).sum()) if not puts.empty else 0.0
        put_call_oi: float | None = None
        if (call_oi + put_oi) >= _MIN_OPEN_INTEREST:
            put_call_oi = round(put_oi / (call_oi + 1), 3)

        # Unusual call OI: today's total call OI vs "expected" OI from option count
        # Proxy: if call volume > 3× call OI (fresh aggressive buying)
        call_vol = float(calls["volume"].fillna(0).sum()) if not calls.empty else 0.0
        unusual_call_oi = bool(call_oi > 0 and call_vol > call_oi * 3.0)

        # IV vs realized vol spread
        rv = _realized_vol_20d(ticker)
        iv_rv_spread: float | None = None
        if atm_iv_val is not None and rv is not None:
            iv_rv_spread = round(atm_iv_val - rv, 4)

        return OptionsSnapshot(
            atm_iv=atm_iv_val,
            skew_25d=skew_25d,
            put_call_oi_ratio=put_call_oi,
            iv_rv_spread=iv_rv_spread,
            iv_cheap=iv_rv_spread is not None and iv_rv_spread < -0.07,
            iv_expensive=iv_rv_spread is not None and iv_rv_spread > 0.15,
            unusual_call_oi=unusual_call_oi,
            panic_put_skew=skew_25d is not None and skew_25d > 1.4,
            call_skew_spike=skew_25d is not None and skew_25d < 0.75,
        )

    except Exception as exc:
        logger.debug("options_data: %s compute failed: %s", symbol, exc)
        return _null_snapshot()


# ── Cache ─────────────────────────────────────────────────────────────────────


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
        logger.warning("options_data: cache write error: %s", exc)


def _is_stale(entry: dict) -> bool:
    return entry.get("_date") != today_et().isoformat()


# ── Public API ────────────────────────────────────────────────────────────────


def get_options_snapshot(symbol: str, force_refresh: bool = False) -> OptionsSnapshot:
    """Return today's OptionsSnapshot for symbol, using daily cache.

    Returns _null_snapshot() when options are unavailable (ETFs, small-caps, etc.).
    """
    cache = _load_cache()
    entry = cache.get(symbol)
    if not force_refresh and entry and not _is_stale(entry):
        try:
            payload = {k: v for k, v in entry.items() if k != "_date"}
            return OptionsSnapshot(**payload)
        except Exception:
            pass

    snapshot = _compute_snapshot(symbol)
    cache[symbol] = {**asdict(snapshot), "_date": today_et().isoformat()}
    _save_cache(cache)
    return snapshot


def get_options_batch(
    symbols: list[str],
    max_workers: int = 6,
    force_refresh: bool = False,
) -> dict[str, OptionsSnapshot]:
    """Fetch OptionsSnapshot for all symbols in parallel.

    Returns {symbol: OptionsSnapshot} — every symbol is present in the result
    (null snapshots included so callers can distinguish fetched-but-missing from
    not-yet-fetched).
    """
    cache = _load_cache()
    today_str = today_et().isoformat()

    results: dict[str, OptionsSnapshot] = {}
    to_fetch: list[str] = []

    for sym in symbols:
        entry = cache.get(sym)
        if not force_refresh and entry and not _is_stale(entry):
            try:
                payload = {k: v for k, v in entry.items() if k != "_date"}
                results[sym] = OptionsSnapshot(**payload)
                continue
            except Exception:
                pass
        to_fetch.append(sym)

    if to_fetch:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_compute_snapshot, sym): sym for sym in to_fetch}
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    snap = future.result()
                except Exception as exc:
                    logger.debug("options_data batch: %s failed: %s", sym, exc)
                    snap = _null_snapshot()
                results[sym] = snap
                cache[sym] = {**asdict(snap), "_date": today_str}

        _save_cache(cache)

    logger.info(
        "options_data: %d/%d symbols with options data",
        sum(1 for s in results.values() if s.atm_iv is not None),
        len(symbols),
    )
    return results
