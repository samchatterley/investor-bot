"""Signal × direction matrix backtest — does any signal have edge in the direction it ISN'T classified?

Motivation (this week's thread): we classify each signal as long OR short, but trigger and direction
are independent. A disabled long might be a short (fade-as-is); a disabled short might be a long
(e.g. "buy parabolic_exhaustion" instead of shorting it). This settles every cell at once.

Method — a symmetric forward-return EVENT STUDY (not a portfolio sim, so position caps/stops don't
confound the signal's raw directional edge):
  * Reuse the engine's validated `_row_to_snapshot` + `evaluate_signals` / `evaluate_short_signals`
    (both disabled-sets monkeypatched empty so disabled signals fire).
  * For every (symbol, date) a signal fires, take the fixed-horizon forward close-to-close return.
  * Score BOTH directions from the same event:
       long_excess = fwd_ret - spy_fwd_ret            (beat the market — the drift makes raw longs
                                                        look good, so we require EXCESS)
       short_net   = -fwd_ret - borrow_cost(horizon)  (short profits when it falls, minus borrow)
  * Walk-forward by calendar year; report mean-of-fold-means, fold consistency (% profitable folds),
    n, and the worst fold (tail) — because with ~30 signals × 2 directions some cells WILL look good
    by chance, so we trust consistency + tail, not a single in-sample average.

Caveats: borrow is flat-assumed (no point-in-time fee feed → short `net` is an upper bound);
survivorship bias (today's listings); catalyst/fundamental signals need feeds absent from OHLCV and
simply won't fire here (this study is for the technical/price book, which is exactly the disabled set).

Usage:
    python scripts/signal_direction_matrix.py [--start 2015-01-01] [--end 2023-12-31]
        [--limit N] [--horizon 5] [--borrow-annual 3.0] [--min-trades 50] [--min-consistency 60]
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import signals.evaluator as _ev  # noqa: E402
from backtest.engine import STOCK_UNIVERSE as _UNIVERSE  # noqa: E402
from backtest.engine import _row_to_snapshot  # noqa: E402
from scripts.short_disabled_backtest import _prep  # noqa: E402


def _spy_returns(spy_df, date):
    """(ret_5d, ret_10d) for SPY ending at `date`, or (None, None)."""
    if spy_df is None or date not in spy_df.index:
        return None, None
    loc = spy_df.index.get_loc(date)

    def _r(n):
        if loc < n:
            return None
        a, b = float(spy_df["Close"].iloc[loc - n]), float(spy_df["Close"].iloc[loc])
        return (b / a - 1) * 100 if a > 0 else None

    return _r(5), _r(10)


def _scan(prep: dict, horizon: int) -> tuple[dict, dict]:
    """Return (events, native_dir): events[signal] = [(year, fwd_ret_pct, spy_fwd_ret_pct)]."""
    indicators = prep["indicators"]
    spy_df = prep["sim_kwargs"]["spy_indicators"]
    rs_ranks = prep["sim_kwargs"]["rs_ranks"] or {}
    events: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    native_dir: dict[str, str] = {}

    spy_close = spy_df["Close"] if spy_df is not None else None
    for sym, df in indicators.items():
        if sym == "SPY" or "Close" not in df.columns:
            continue
        closes = df["Close"]
        n = len(df)
        sym_ranks = rs_ranks.get(sym, {})
        for i in range(n - horizon):
            date = df.index[i]
            entry = float(closes.iloc[i])
            exit_ = float(closes.iloc[i + horizon])
            if not (math.isfinite(entry) and math.isfinite(exit_)) or entry <= 0:
                continue
            fwd = (exit_ / entry - 1) * 100
            # market move over the same window (for long excess + as a beta proxy for short)
            spy_fwd = 0.0
            if spy_close is not None and date in spy_close.index:
                sloc = spy_close.index.get_loc(date)
                if sloc + horizon < len(spy_close):
                    a, b = float(spy_close.iloc[sloc]), float(spy_close.iloc[sloc + horizon])
                    spy_fwd = (b / a - 1) * 100 if a > 0 else 0.0
            try:
                s5, s10 = _spy_returns(spy_df, date)
                snap = _row_to_snapshot(df.iloc[i], spy_ret_5d=s5, spy_ret_10d=s10)
                snap["rs_rank_pct"] = sym_ranks.get(date.strftime("%Y-%m-%d"), 50.0)
                longs = _ev.evaluate_signals(snap, spy_ret_5d=s5, spy_ret_10d=s10)
                shorts = _ev.evaluate_short_signals(snap)
            except Exception:
                continue
            yr = date.year
            for s in longs:
                events[s].append((yr, fwd, spy_fwd))
                native_dir.setdefault(s, "long")
            for s in shorts:
                events[s].append((yr, fwd, spy_fwd))
                native_dir.setdefault(s, "short")
    return events, native_dir


def _fold_stats(per_year: dict[int, list[float]]) -> tuple[float, float, float, int]:
    """(mean of per-fold means, % profitable folds, worst fold mean, n_trades)."""
    fold_means = [statistics.mean(v) for v in per_year.values() if v]
    n = sum(len(v) for v in per_year.values())
    if not fold_means:
        return 0.0, 0.0, 0.0, n
    consistency = sum(1 for m in fold_means if m > 0) / len(fold_means) * 100
    return statistics.mean(fold_means), consistency, min(fold_means), n


def main() -> None:
    ap = argparse.ArgumentParser(description="Signal × direction matrix event study")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2023-12-31")  # keep < holdout (2024-01-01)
    ap.add_argument("--limit", type=int, default=120, help="cap universe (0 = full)")
    ap.add_argument("--horizon", type=int, default=5, help="forward holding days")
    ap.add_argument("--borrow-annual", type=float, default=3.0, help="flat borrow %/yr for shorts")
    ap.add_argument("--min-trades", type=int, default=50)
    ap.add_argument("--min-consistency", type=float, default=60.0, help="min %% profitable folds")
    args = ap.parse_args()

    universe = list(_UNIVERSE)
    if args.limit:
        universe = universe[: args.limit]
        if "SPY" not in universe:
            universe.append("SPY")

    print("=" * 96)
    print(f"  SIGNAL × DIRECTION MATRIX  {args.start} → {args.end}  horizon={args.horizon}d")
    print(
        f"  universe={len(universe)}  borrow={args.borrow_annual}%/yr  bars: long=EXCESS vs SPY, short=NET"
    )
    print("=" * 96)

    prep = _prep(universe, args.start, args.end, with_earnings=False)

    # Un-disable everything so disabled signals fire in the study.
    orig_long, orig_short = _ev.GLOBALLY_DISABLED, _ev.SHORT_GLOBALLY_DISABLED
    _ev.GLOBALLY_DISABLED = frozenset()
    _ev.SHORT_GLOBALLY_DISABLED = frozenset()
    try:
        events, native_dir = _scan(prep, args.horizon)
    finally:
        _ev.GLOBALLY_DISABLED, _ev.SHORT_GLOBALLY_DISABLED = orig_long, orig_short

    borrow_pct = args.borrow_annual * args.horizon / 252.0
    disabled = set(orig_long) | set(orig_short)

    rows = []
    for sig, evs in events.items():
        long_by_year: dict[int, list[float]] = defaultdict(list)
        short_by_year: dict[int, list[float]] = defaultdict(list)
        for yr, fwd, spy_fwd in evs:
            long_by_year[yr].append(fwd - spy_fwd)  # excess long
            short_by_year[yr].append(-fwd - borrow_pct)  # net short
        l_mean, l_cons, l_worst, n = _fold_stats(long_by_year)
        s_mean, s_cons, s_worst, _ = _fold_stats(short_by_year)
        rows.append(
            (sig, native_dir.get(sig, "?"), n, l_mean, l_cons, l_worst, s_mean, s_cons, s_worst)
        )

    rows.sort(key=lambda r: r[2], reverse=True)

    def _flag(mean, cons, worst, n):
        if n < args.min_trades:
            return " "
        return (
            "*" if (mean > 0 and cons >= args.min_consistency and worst > -2 * abs(mean)) else " "
        )

    hdr = f"  {'signal':<28}{'nat':>5}{'n':>6}   {'LONG ex%':>9}{'fold+':>6}{'worst':>7}   {'SHORT net%':>11}{'fold+':>6}{'worst':>7}"
    print("\n" + hdr)
    print("  " + "-" * 92)
    flips = []
    for sig, nat, n, lm, lc, lw, sm, sc, sw in rows:
        lf, sf = _flag(lm, lc, lw, n), _flag(sm, sc, sw, n)
        print(
            f"  {sig:<28}{nat:>5}{n:>6}   {lm:>8.2f}{lf}{lc:>5.0f}%{lw:>7.2f}   {sm:>10.2f}{sf}{sc:>5.0f}%{sw:>7.2f}"
        )
        # A "flip" = a signal that looks good in the direction OPPOSITE its classification.
        if nat == "long" and sf == "*":
            flips.append(
                f"{sig}: SHORT net {sm:+.2f}%/{sc:.0f}% folds (classified long{' , DISABLED' if sig in disabled else ''})"
            )
        if nat == "short" and lf == "*":
            flips.append(
                f"{sig}: LONG excess {lm:+.2f}%/{lc:.0f}% folds (classified short{' , DISABLED' if sig in disabled else ''})"
            )
    print("  " + "-" * 92)
    print(
        f"  * = n>={args.min_trades}, positive mean, >={args.min_consistency:.0f}% folds profitable, no catastrophic fold"
    )
    print("\n  CROSS-CLASSIFICATION CANDIDATES (edge in the non-native direction):")
    for f in flips:
        print(f"    • {f}")
    if not flips:
        print("    none — no signal shows robust edge in its non-native direction")
    print("\n  Reminder: borrow flat-assumed (short net = upper bound); event study (no stops);")
    print(
        "  multiple cells tested → trust fold consistency + tail, and re-validate any hit out-of-sample."
    )


if __name__ == "__main__":
    main()
