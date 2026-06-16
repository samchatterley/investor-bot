"""Out-of-sample edge check across S&P cap tiers (large / mid / small).

Question: does the deterministic engine's edge hold on mid (S&P 400) and small (S&P 600) caps as it
does on large (S&P 500)? This decides whether widening the live trading universe to the S&P 1500 is
safe for P&L — the engine was tuned on large caps.

Two passes per tier, both on the pre-holdout window so the 2024+ holdout stays pristine:

  RAW signal edge   — backtest.engine._entry_signal fired bar-by-bar with *no* portfolio filters
                      (no RS-rank, regime, quality screen, or costs). Measures the bare signal book.
  FULL pipeline     — backtest.engine.run_backtest: the real simulator with cross-sectional RS-rank,
                      SPY/VIX regime blocks, fundamental quality screens, the cost model, sizing, and
                      max-positions. This is what actually trades. If the live filters rescue the
                      down-cap names, the FULL win-rate / avg-return-per-trade recovers even where the
                      RAW edge is negative.

Run:  python scripts/universe_edge_check.py
"""

from __future__ import annotations  # pragma: no cover

import logging  # pragma: no cover
import math  # pragma: no cover
import os  # pragma: no cover
import pickle  # pragma: no cover
import random  # pragma: no cover
import sys  # pragma: no cover
import warnings  # pragma: no cover

warnings.filterwarnings("ignore")  # pragma: no cover
logging.disable(logging.CRITICAL)  # pragma: no cover
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # pragma: no cover

from backtest.engine import (  # noqa: E402  # pragma: no cover
    _compute_indicators,
    _entry_signal,
    run_backtest,
)
from data.market_data import _download_symbols  # noqa: E402  # pragma: no cover

_N_PER_TIER = 30  # pragma: no cover
_START = "2021-01-01"  # pre-holdout window (holdout starts 2024-01-01)  # pragma: no cover
_END = "2023-12-31"  # pragma: no cover
_FWD = 5  # raw event-study forward horizon  # pragma: no cover


def _sample_tiers() -> dict[str, list[str]]:  # pragma: no cover
    with open("/tmp/universes.pkl", "rb") as fh:
        d = pickle.load(fh)
    random.seed(42)
    out = {}
    for label, key in (("large", "sp500"), ("mid", "sp400"), ("small", "sp600")):
        plain = sorted(x for x in d[key] if x.isalpha())
        out[label] = random.sample(plain, min(_N_PER_TIER, len(plain)))
    return out


def _raw_edge(syms: list[str]) -> dict:  # pragma: no cover
    """Bare signal-book edge: mean(fwd | signal) - mean(fwd | all bars), no filters."""
    data = _download_symbols(syms, 1300)
    all_ret: list[float] = []
    sig_ret: list[float] = []
    for df in data.values():
        if len(df) < 220:
            continue
        df = _compute_indicators(df)
        closes = df["Close"].to_numpy()
        for i in range(200, len(df) - _FWD):
            fr = closes[i + _FWD] / closes[i] - 1.0
            if not math.isfinite(fr):
                continue
            all_ret.append(fr)
            try:
                if _entry_signal(df.iloc[i]):
                    sig_ret.append(fr)
            except Exception:
                continue
    base = sum(all_ret) / len(all_ret) if all_ret else 0.0
    sig = sum(sig_ret) / len(sig_ret) if sig_ret else 0.0
    win = sum(1 for r in sig_ret if r > 0) / len(sig_ret) * 100 if sig_ret else 0.0
    return {"n_sig": len(sig_ret), "edge_pct": (sig - base) * 100, "win_pct": win}


def main() -> None:  # pragma: no cover
    tiers = _sample_tiers()

    print(f"\nRAW signal edge (no filters), {_FWD}-day forward")
    print(f"{'tier':<8}{'sig bars':>9}{'EDGE%':>9}{'win%':>7}")
    raw = {}
    for label, syms in tiers.items():
        raw[label] = _raw_edge(syms)
        r = raw[label]
        print(f"{label:<8}{r['n_sig']:>9}{r['edge_pct']:>8.2f}%{r['win_pct']:>6.0f}%")

    print(f"\nFULL pipeline (RS-rank + regime + quality + costs), {_START}..{_END}")
    print(f"{'tier':<8}{'trades':>7}{'win%':>7}{'avg/trade%':>12}{'total%':>9}{'sharpe':>8}")
    for label, syms in tiers.items():
        res = run_backtest(syms, _START, _END, use_fundamentals=True, max_positions=10)
        if not res:
            print(f"{label:<8}  (no result)")
            continue
        print(
            f"{label:<8}{res.get('total_trades', 0):>7}{res.get('win_rate_pct', 0):>6.0f}%"
            f"{res.get('avg_return_per_trade_pct', 0):>11.2f}%{res.get('total_return_pct', 0):>8.1f}%"
            f"{res.get('sharpe_ratio', 0):>8.2f}"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
