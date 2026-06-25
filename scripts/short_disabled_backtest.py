"""ADR-006 B4 — isolated backtest of the disabled short signals.

Research, NOT a re-enable. For each name in ``SHORT_GLOBALLY_DISABLED`` this temporarily
un-disables *only that signal*, runs the short-side simulation in isolation
(``signals_only={sig}``) over a fixed pre-holdout window, and reports trades / win-rate /
avg return / Sharpe so we can decide — per the CLAUDE.md disable/enable checklist — whether
any are worth a properly-gated re-enable.

It reuses the engine's validated pipeline (`_run_short_simulation`) and the same data-prep
as `run_short_ablation`, downloading the universe once and reusing it across all signals.

Caveats (read before trusting any "RE-ENABLE?" verdict):
  * Borrow cost is NOT modelled → fundamental/HTB shorts read optimistically.
  * Survivorship bias: the universe is today's listings.
  * `altman_distress_short` / `gross_margin_deterioration_short` need point-in-time quality
    fundamentals, which `_run_short_simulation` does not receive here → expect 0 trades;
    they must be evaluated via the quality-fundamentals combined path separately.
  * `iv_compression_short` uses an HV-proxy in the backtest (no historical IV), so its number
    is a proxy, not the live signal.

Usage:
    python scripts/short_disabled_backtest.py [--start 2015-01-01] [--end 2023-12-31]
                                              [--limit N] [--max-positions 2]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

import signals.evaluator as _ev  # noqa: E402
from backtest.engine import (  # noqa: E402
    _SHORT_MAX_HOLD_DAYS,
    _build_indicators,
    _build_regime_map,
    _compute_rs_rank_lag10,
    _compute_rs_ranks,
    _run_short_simulation,
    prefetch_earnings_history,
)
from backtest.engine import STOCK_UNIVERSE as _UNIVERSE  # noqa: E402

# Re-enable bar (research screen — all three must hold; borrow cost not modelled so these
# are a NECESSARY-not-sufficient screen, deliberately lenient on n given how rare shorts are):
MIN_TRADES = 30  # below this the expectancy estimate is noise
MIN_AVG_PCT = 0.0  # must be net-positive per trade (pre-borrow-cost)
MIN_SHARPE = 0.30  # a meaningful risk-adjusted edge, not a coin flip


def _prep(symbols: list[str], start: str, end: str, *, with_earnings: bool = True) -> dict:
    """Download + build everything the short sim needs (mirrors run_short_ablation prep).

    with_earnings=False skips prefetch_earnings_history (yfinance's per-symbol earnings endpoint
    is heavily throttled); earnings_miss / faded_earnings_gap_up then take no trades and must be
    evaluated in a dedicated earnings-history run.
    """
    from datetime import datetime, timedelta

    fetch_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
    print(f"Downloading {len(symbols)} symbols {fetch_start} → {end} …", flush=True)
    raw = yf.download(symbols, start=fetch_start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise SystemExit("No data fetched — check network / yfinance availability.")

    indicators = _build_indicators(raw, symbols)
    spy_indicators = indicators.get("SPY")

    vix_df: pd.DataFrame | None = None
    try:
        vix_raw = yf.download("^VIX", start=fetch_start, end=end, auto_adjust=False, progress=False)
        if not vix_raw.empty:
            vix_close = vix_raw["Close"]
            if isinstance(vix_close, pd.DataFrame):
                vix_close = vix_close.iloc[:, 0]
            vix_df = pd.DataFrame({"Close": vix_close.astype(float)}).dropna()
            vix_df.index = pd.DatetimeIndex(vix_df.index).tz_localize(None)
    except Exception as exc:
        print(f"  VIX fetch failed ({exc}) — regime map degraded")

    regime_by_date: dict[str, str] = {}
    if spy_indicators is not None and "Close" in spy_indicators.columns:
        regime_by_date = _build_regime_map(spy_indicators, vix_df)

    rs_ranks = _compute_rs_ranks(indicators, spy_indicators)
    trading_dates = pd.bdate_range(start=start, end=end)
    rs_rank_lag10 = _compute_rs_rank_lag10(rs_ranks, trading_dates)
    if with_earnings:
        print("Prefetching earnings history (slow — yfinance per-symbol) …", flush=True)
        earnings_history = prefetch_earnings_history(symbols)
    else:
        print("Skipping earnings prefetch (--no-earnings) …", flush=True)
        earnings_history = None

    return {
        "indicators": indicators,
        "trading_dates": trading_dates,
        "sim_kwargs": {
            "spy_indicators": spy_indicators,
            "regime_by_date": regime_by_date or None,
            "rs_ranks": rs_ranks,
            "rs_rank_lag10": rs_rank_lag10,
            "earnings_history": earnings_history,
        },
    }


def _run_one(sig: str, prep: dict, initial_capital: float, max_positions: int) -> dict:
    """Un-disable `sig` only, run the short sim isolated to it, restore. Returns its stats."""
    original = _ev.SHORT_GLOBALLY_DISABLED
    _ev.SHORT_GLOBALLY_DISABLED = original - {sig}
    try:
        res = _run_short_simulation(
            prep["indicators"],
            prep["trading_dates"],
            initial_capital=initial_capital,
            max_positions=max_positions,
            max_hold_days=_SHORT_MAX_HOLD_DAYS,
            signals_only=frozenset({sig}),
            **prep["sim_kwargs"],
        )
    finally:
        _ev.SHORT_GLOBALLY_DISABLED = original
    return res


def _verdict(n: int, avg: float, sharpe: float) -> str:
    if n < MIN_TRADES:
        return f"KEEP (n<{MIN_TRADES})"
    if avg > MIN_AVG_PCT and sharpe >= MIN_SHARPE:
        return "RE-ENABLE?"
    return "KEEP"


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest the disabled short signals (ADR-006 B4)")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2023-12-31")  # must stay < HOLDOUT_START_DATE (2024-01-01)
    ap.add_argument("--capital", type=float, default=25_000.0)
    ap.add_argument("--max-positions", type=int, default=2)
    ap.add_argument(
        "--limit", type=int, default=0, help="cap universe size (0 = full) for a quick run"
    )
    ap.add_argument(
        "--no-earnings",
        action="store_true",
        help="skip the (throttled) earnings prefetch; earnings_miss/faded_earnings_gap_up take 0 trades",
    )
    ap.add_argument(
        "--signals",
        default="",
        help="comma list of signals to isolate instead of the disabled set (e.g. active shorts). "
        "Works for active or disabled names — disabled ones are un-disabled for the run.",
    )
    args = ap.parse_args()

    universe = list(_UNIVERSE)
    if args.limit:
        # keep SPY for regime/RS computation regardless of the cap
        head = universe[: args.limit]
        if "SPY" not in head:
            head.append("SPY")
        universe = head

    if args.signals:
        targets = [s.strip() for s in args.signals.split(",") if s.strip()]
        unknown = [s for s in targets if s not in _ev.SHORT_SIGNAL_PRIORITY]
        if unknown:
            raise SystemExit(f"Unknown short signal(s): {unknown}")
        disabled = sorted(targets, key=lambda s: _ev.SHORT_SIGNAL_PRIORITY.get(s, 99))
    else:
        disabled = sorted(
            _ev.SHORT_GLOBALLY_DISABLED, key=lambda s: _ev.SHORT_SIGNAL_PRIORITY.get(s, 99)
        )
    print("=" * 78)
    print(f"  ADR-006 B4 — disabled-short isolation backtest  {args.start} → {args.end}")
    print(
        f"  universe={len(universe)}  capital=${args.capital:,.0f}  max_short={args.max_positions}"
    )
    print(f"  signals={len(disabled)}: {', '.join(disabled)}")
    print("=" * 78)

    prep = _prep(universe, args.start, args.end, with_earnings=not args.no_earnings)

    rows = []
    for sig in disabled:
        print(f"  running {sig} …", flush=True)
        r = _run_one(sig, prep, args.capital, args.max_positions)
        rows.append(
            {
                "signal": sig,
                "trades": r.get("total_trades", 0),
                "win_rate": r.get("win_rate_pct", 0.0),
                "avg": r.get("avg_return_per_trade_pct", 0.0),
                "sharpe": r.get("sharpe_ratio", 0.0),
                "total_return": r.get("total_return_pct", 0.0),
            }
        )

    rows.sort(key=lambda x: (x["sharpe"], x["avg"]), reverse=True)
    print("\n" + "=" * 78)
    print(f"  {'Signal':<32}{'Trades':>7}{'WR%':>7}{'Avg%':>8}{'Sharpe':>8}  Verdict")
    print("  " + "-" * 74)
    for x in rows:
        v = _verdict(x["trades"], x["avg"], x["sharpe"])
        print(
            f"  {x['signal']:<32}{x['trades']:>7}{x['win_rate']:>7.1f}"
            f"{x['avg']:>8.2f}{x['sharpe']:>8.2f}  {v}"
        )
    print("  " + "-" * 74)
    cands = [
        x["signal"] for x in rows if _verdict(x["trades"], x["avg"], x["sharpe"]) == "RE-ENABLE?"
    ]
    print(
        f"\n  RE-ENABLE candidates (research screen only, borrow cost NOT modelled): "
        f"{cands or 'none'}"
    )
    print("  Any candidate must still pass the CLAUDE.md disable/enable checklist + borrow-cost")
    print("  realism + a walk-forward / out-of-sample check before it ships.")


if __name__ == "__main__":
    main()
