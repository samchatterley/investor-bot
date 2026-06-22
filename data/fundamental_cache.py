"""Fundamental data cache — weekly yfinance fetch for all 509 symbols.

Provides per-symbol: Piotroski F-score, Altman Z-score, TTM FCF yield,
quarterly gross margin trend, forward P/E, shares outstanding, market cap.

Cache lives at logs/fundamental_cache.json and is refreshed per symbol when
stale (>7 days). Call refresh_fundamental_cache() from the weekly prefetch
job to warm all 509 symbols up front.

All public getters return None on missing data rather than raising.
"""

import json
import logging
import os
from datetime import date
from typing import Any

import pandas as pd
import yfinance as yf

from config import LOG_DIR, STOCK_UNIVERSE, today_et

logger = logging.getLogger(__name__)

_CACHE_PATH = os.path.join(LOG_DIR, "caching", "fundamental_cache.json")
_CACHE_TTL_DAYS = 7


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
        logger.warning(f"fundamental_cache: write error: {e}")


def _is_stale(entry: dict) -> bool:
    try:
        last = date.fromisoformat(entry["last_updated"])
        return (today_et() - last).days > _CACHE_TTL_DAYS
    except (KeyError, ValueError):
        return True


# ── DataFrame helpers ─────────────────────────────────────────────────────────


def _row(df: pd.DataFrame, *names: str) -> pd.Series | None:
    """Return first matching row from df by trying each name in order."""
    for name in names:
        if name in df.index:
            return df.loc[name]
    return None


def _val(series: pd.Series | None, col: int = 0) -> float | None:
    """Return series.iloc[col] as float, or None if missing/NaN."""
    if series is None or len(series) == 0:
        return None
    try:
        v = series.iloc[col]
        return float(v) if pd.notna(v) else None
    except (IndexError, TypeError, ValueError):
        return None


# ── Piotroski F-score (0–9) ───────────────────────────────────────────────────


def _compute_piotroski(ticker: yf.Ticker) -> int | None:
    """Compute Piotroski F-score from annual financial statements.

    9 binary criteria: profitability (4), leverage/liquidity (3), efficiency (2).
    Returns None when data is insufficient.
    """
    try:
        fin = ticker.financials
        bs = ticker.balance_sheet
        cf = ticker.cashflow

        if fin.empty or bs.empty or cf.empty:
            return None
        if fin.shape[1] < 2 or bs.shape[1] < 2:
            return None

        total_assets_cur = _val(_row(bs, "Total Assets"), 0)
        total_assets_pri = _val(_row(bs, "Total Assets"), 1)
        if not total_assets_cur or not total_assets_pri:
            return None

        net_income_cur = _val(_row(fin, "Net Income"), 0)
        net_income_pri = _val(_row(fin, "Net Income"), 1)
        ocf_cur = _val(_row(cf, "Operating Cash Flow"), 0)
        lt_debt_cur = _val(_row(bs, "Long Term Debt"), 0)
        lt_debt_pri = _val(_row(bs, "Long Term Debt"), 1)
        cur_assets_cur = _val(_row(bs, "Current Assets"), 0)
        cur_liab_cur = _val(_row(bs, "Current Liabilities"), 0)
        cur_assets_pri = _val(_row(bs, "Current Assets"), 1)
        cur_liab_pri = _val(_row(bs, "Current Liabilities"), 1)
        shares_cur = _val(_row(bs, "Share Issued", "Ordinary Shares Number"), 0)
        shares_pri = _val(_row(bs, "Share Issued", "Ordinary Shares Number"), 1)
        gross_profit_cur = _val(_row(fin, "Gross Profit"), 0)
        gross_profit_pri = _val(_row(fin, "Gross Profit"), 1)
        revenue_cur = _val(_row(fin, "Total Revenue"), 0)
        revenue_pri = _val(_row(fin, "Total Revenue"), 1)

        score = 0

        # F1: ROA > 0
        if net_income_cur is not None and net_income_cur / total_assets_cur > 0:
            score += 1

        # F2: Operating cash flow > 0
        if ocf_cur is not None and ocf_cur > 0:
            score += 1

        # F3: ΔROA > 0 (profitability improving)
        if (
            net_income_cur is not None
            and net_income_pri is not None
            and net_income_cur / total_assets_cur > net_income_pri / total_assets_pri
        ):
            score += 1

        # F4: Accruals quality (OCF/assets > ROA means cash earnings exceed accrual)
        if (
            ocf_cur is not None
            and net_income_cur is not None
            and ocf_cur / total_assets_cur > net_income_cur / total_assets_cur
        ):
            score += 1

        # F5: ΔLeverage < 0 (debt ratio decreased)
        if (
            lt_debt_cur is not None
            and lt_debt_pri is not None
            and lt_debt_cur / total_assets_cur < lt_debt_pri / total_assets_pri
        ):
            score += 1

        # F6: ΔLiquidity > 0 (current ratio improved)
        if (
            cur_assets_cur is not None
            and cur_liab_cur is not None
            and cur_assets_pri is not None
            and cur_liab_pri is not None
            and cur_liab_cur != 0
            and cur_liab_pri != 0
            and cur_assets_cur / cur_liab_cur > cur_assets_pri / cur_liab_pri
        ):
            score += 1

        # F7: No material dilution (shares not up more than 2%)
        if (
            shares_cur is not None
            and shares_pri is not None
            and shares_pri != 0
            and shares_cur <= shares_pri * 1.02
        ):
            score += 1

        # F8: ΔGross margin > 0
        if (
            gross_profit_cur is not None
            and gross_profit_pri is not None
            and revenue_cur is not None
            and revenue_pri is not None
            and revenue_cur != 0
            and revenue_pri != 0
            and gross_profit_cur / revenue_cur > gross_profit_pri / revenue_pri
        ):
            score += 1

        # F9: ΔAsset turnover > 0
        if (
            revenue_cur is not None
            and revenue_pri is not None
            and revenue_cur / total_assets_cur > revenue_pri / total_assets_pri
        ):
            score += 1

        return score

    except Exception as e:
        logger.debug(f"piotroski: {e}")
        return None


# ── Altman Z-score ────────────────────────────────────────────────────────────


def _compute_altman_z(ticker: yf.Ticker) -> float | None:
    """Compute Altman Z-score (1968 public-company model).

    Z > 2.6 = safe; 1.1–2.6 = grey zone; < 1.1 = distress.
    Returns None when any required input is unavailable.
    """
    try:
        bs = ticker.balance_sheet
        fin = ticker.financials
        info = ticker.info or {}

        if bs.empty or fin.empty:
            return None

        total_assets = _val(_row(bs, "Total Assets"), 0)
        if not total_assets:
            return None

        working_capital = _val(_row(bs, "Working Capital"), 0)
        retained_earnings = _val(_row(bs, "Retained Earnings"), 0)
        ebit = _val(_row(fin, "EBIT"), 0)
        revenue = _val(_row(fin, "Total Revenue"), 0)
        total_liab = _val(_row(bs, "Total Liabilities Net Minority Interest"), 0)
        _mc = info.get("marketCap")
        market_cap = float(_mc) if isinstance(_mc, (int, float)) else None

        # Explicit per-value None checks (not any()) so the type checker can narrow each to
        # float for the arithmetic below.
        if (
            working_capital is None
            or retained_earnings is None
            or ebit is None
            or revenue is None
            or total_liab is None
            or market_cap is None
            or total_liab == 0
        ):
            return None

        x1 = working_capital / total_assets
        x2 = retained_earnings / total_assets
        x3 = ebit / total_assets
        x4 = market_cap / total_liab
        x5 = revenue / total_assets

        return round(1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5, 3)

    except Exception as e:
        logger.debug(f"altman_z: {e}")
        return None


# ── TTM FCF yield ─────────────────────────────────────────────────────────────


def _compute_fcf_yield(ticker: yf.Ticker) -> float | None:
    """Compute TTM FCF yield = sum-of-4-quarterly-FCF / market_cap.

    Falls back to annual FCF when quarterly data has fewer than 4 periods.
    Returns None when market_cap or FCF is unavailable.
    """
    try:
        info = ticker.info or {}
        market_cap = info.get("marketCap")
        if not market_cap:
            return None

        qcf = ticker.quarterly_cashflow
        fcf_series = _row(qcf, "Free Cash Flow")

        if fcf_series is not None and len(fcf_series) >= 4:
            values = [fcf_series.iloc[i] for i in range(4)]
            if any(pd.isna(v) for v in values):
                return None
            fcf_ttm = sum(float(v) for v in values)
        else:
            # Fewer than 4 quarters — fall back to most recent annual figure
            cf = ticker.cashflow
            if cf.empty:
                return None
            fcf_annual = _val(_row(cf, "Free Cash Flow"), 0)
            if fcf_annual is None:
                return None
            fcf_ttm = fcf_annual

        return round(fcf_ttm / market_cap, 4)

    except Exception as e:
        logger.debug(f"fcf_yield: {e}")
        return None


# ── Gross margin trend ────────────────────────────────────────────────────────


def _compute_gross_margin_trend(
    ticker: yf.Ticker,
) -> tuple[float | None, float | None, float | None]:
    """Compute gross margin: (current_quarter, 4q_avg, trend_delta).

    trend_delta > 0 means the current quarter margin is above the 4-quarter
    rolling average (improving). Returns (None, None, None) when insufficient
    data.
    """
    try:
        qfin = ticker.quarterly_financials
        gp = _row(qfin, "Gross Profit")
        rev = _row(qfin, "Total Revenue")

        if gp is None or rev is None or len(gp) < 4 or len(rev) < 4:
            return None, None, None

        margins: list[float | None] = []
        for i in range(min(5, len(gp), len(rev))):
            gp_v = float(gp.iloc[i]) if pd.notna(gp.iloc[i]) else None
            rv_v = float(rev.iloc[i]) if pd.notna(rev.iloc[i]) else None
            if gp_v is None or rv_v is None or rv_v == 0:
                margins.append(None)
            else:
                margins.append(gp_v / rv_v)

        valid = [m for m in margins if m is not None]
        if len(valid) < 4:
            return None, None, None

        current = valid[0]
        avg_4q = sum(valid[:4]) / 4
        return round(current, 4), round(avg_4q, 4), round(current - avg_4q, 4)

    except Exception as e:
        logger.debug(f"gross_margin_trend: {e}")
        return None, None, None


# ── Accruals quality ─────────────────────────────────────────────────────────


def _compute_accruals_ratio(ticker: yf.Ticker) -> float | None:
    """Compute accruals ratio = (net_income - ocf) / total_assets.

    Sloan (1996): high accruals predict earnings reversals.
    Positive value = accruals exceed cash earnings (earnings quality risk).
    Returns None when required data is unavailable.
    """
    try:
        fin = ticker.financials
        bs = ticker.balance_sheet
        cf = ticker.cashflow
        if fin.empty or bs.empty or cf.empty:
            return None
        total_assets = _val(_row(bs, "Total Assets"), 0)
        if not total_assets:
            return None
        net_income = _val(_row(fin, "Net Income"), 0)
        ocf = _val(_row(cf, "Operating Cash Flow"), 0)
        if net_income is None or ocf is None:
            return None
        return round((net_income - ocf) / total_assets, 4)
    except Exception as e:
        logger.debug(f"accruals_ratio: {e}")
        return None


# ── Single-symbol fetch ───────────────────────────────────────────────────────


def _fetch_symbol(sym: str) -> dict:
    """Fetch all fundamental fields for sym from yfinance. Returns {} on failure."""
    try:
        t = yf.Ticker(sym)
        info = t.info or {}
        gm_cur, gm_4q, gm_trend = _compute_gross_margin_trend(t)
        return {
            "piotroski_f": _compute_piotroski(t),
            "altman_z": _compute_altman_z(t),
            "fcf_yield": _compute_fcf_yield(t),
            "gross_margin_current": gm_cur,
            "gross_margin_4q_avg": gm_4q,
            "gross_margin_trend": gm_trend,
            "accruals_ratio": _compute_accruals_ratio(t),
            "forward_pe": info.get("forwardPE"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "market_cap": info.get("marketCap"),
            "last_updated": today_et().isoformat(),
        }
    except Exception as e:
        logger.debug(f"fundamental_cache: fetch {sym}: {e}")
        return {}


# ── Public refresh ────────────────────────────────────────────────────────────


def refresh_fundamental_cache(
    symbols: list[str] | None = None,
    force: bool = False,
) -> int:
    """Refresh stale entries in the fundamental cache.

    Args:
        symbols: symbols to check; defaults to STOCK_UNIVERSE.
        force: if True, refresh all symbols regardless of age.

    Returns:
        Number of symbols refreshed.
    """
    if symbols is None:
        symbols = list(STOCK_UNIVERSE)

    cache = _load_cache()
    refreshed = 0

    for sym in symbols:
        entry = cache.get(sym, {})
        if not force and entry and not _is_stale(entry):
            continue
        data = _fetch_symbol(sym)
        if data:
            cache[sym] = data
            refreshed += 1
            logger.debug(f"fundamental_cache: refreshed {sym}")

    if refreshed:
        _save_cache(cache)
        logger.info(f"fundamental_cache: refreshed {refreshed}/{len(symbols)} symbols")

    return refreshed


# ── Per-field getter ──────────────────────────────────────────────────────────


def _get_field(sym: str, field: str) -> Any:
    """Return a cached field for sym, fetching if missing or stale.

    Values come from a JSON cache (numeric, dynamically typed), so the return is ``Any``;
    each public getter coerces to its declared type after a None check.
    """
    cache = _load_cache()
    entry = cache.get(sym)
    if entry is None or _is_stale(entry):
        data = _fetch_symbol(sym)
        if data:
            cache[sym] = data
            _save_cache(cache)
            entry = data
    if entry is None:
        return None
    return entry.get(field)


# ── Public getters ────────────────────────────────────────────────────────────


def get_piotroski_f(sym: str) -> int | None:
    """Return Piotroski F-score (0–9) for sym, or None if unavailable."""
    v = _get_field(sym, "piotroski_f")
    return int(v) if v is not None else None


def get_altman_z(sym: str) -> float | None:
    """Return Altman Z-score for sym, or None if unavailable.

    Interpretation: >2.6 safe, 1.1–2.6 grey, <1.1 distress.
    """
    v = _get_field(sym, "altman_z")
    return float(v) if v is not None else None


def get_fcf_yield(sym: str) -> float | None:
    """Return TTM FCF yield (FCF / market_cap) for sym, or None if unavailable."""
    v = _get_field(sym, "fcf_yield")
    return float(v) if v is not None else None


def get_gross_margin_trend(sym: str) -> float | None:
    """Return gross margin trend delta (current quarter - 4q average).

    Positive = improving, negative = deteriorating, None = unavailable.
    """
    v = _get_field(sym, "gross_margin_trend")
    return float(v) if v is not None else None


def get_gross_margin_current(sym: str) -> float | None:
    """Return most recent quarterly gross margin (0.0–1.0) for sym."""
    v = _get_field(sym, "gross_margin_current")
    return float(v) if v is not None else None


def get_forward_pe(sym: str) -> float | None:
    """Return forward P/E for sym, or None if unavailable."""
    v = _get_field(sym, "forward_pe")
    return float(v) if v is not None else None


def get_shares_outstanding(sym: str) -> int | None:
    """Return shares outstanding for sym, or None if unavailable."""
    v = _get_field(sym, "shares_outstanding")
    return int(v) if v is not None else None


def get_market_cap(sym: str) -> float | None:
    """Return market cap for sym, or None if unavailable."""
    v = _get_field(sym, "market_cap")
    return float(v) if v is not None else None


def get_accruals_ratio(sym: str) -> float | None:
    """Return accruals ratio = (net_income - ocf) / total_assets, or None if unavailable.

    Sloan (1996): high accruals (>0.10) predict earnings reversals; suppress longs.
    """
    v = _get_field(sym, "accruals_ratio")
    return float(v) if v is not None else None
