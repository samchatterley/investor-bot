"""Workshop v2 #5 — peer_lead_lag: the sector moved, this name didn't (yet).

Lo-MacKinlay lead-lag / gradual information diffusion: common-factor news prices into the most-traded
names first; laggards catch up. Thesis: when a name's SECTOR PEERS (equal-weight, ex-self, market-
excess 5d) are up >= +2.5% while the name itself is flat (|me5| < 2%), the laggard catches up over
2-3 sessions. Prior is LOW — info diffuses fast in liquid large caps — but it's a genuinely untested
mechanism family here, price+sector data only, and disjoint from reversal (no own-price drawdown).

Weekly rebalance, entry t+1, hold 3, winsorised excess vs SPY, cost swept. Arms:
  laggard    peers_ex_self_5d >= +2.5% AND own me5 in [-2, +2]  (SIGNAL)
  moved      peers_ex_self_5d >= +2.5% AND own me5 >= +2.5%      (control: already caught up)
  flat-sector own me5 in [-2,+2] AND peers < +2.5%               (control: no peer signal)

Ships if laggard is net>0 @7bps with |t|>=2 and beats both controls; else lead-lag is arbitraged
away in this universe (the expected outcome).

VERDICT: KILL (it's sector beta, not idiosyncratic catch-up). Full sample: laggard peers>=4%
net +0.176%/3d @7bps (t=1.94, 10/12 gross yrs) vs moved-with-peers control -0.061% — looked like a
clean catch-up. But 2021+ OOS: laggard clears the bar (+0.218% net @7bps, t=2.06) YET the pure-beta
control (moved-with-peers) earns a comparable +0.119% net @7bps at t=3.45 — MORE significant than the
laggard. The "peers ran +4%" gate is mostly capturing sector momentum; the name-specific catch-up
over-and-above beta is ~+0.10% and not separately significant. Moved-control flips sign full-sample
(-0.06%) -> OOS (+0.12%), so the whole complex is regime-dependent. Fails the pre-registered
"beats both controls / not sector beta" test in the regime where the control is informative. The
2.5% base gate is a clean miss (t=1.70). Do not ship.

Usage: python scripts/peer_lead_lag.py [--hold 3] [--peer-min 2.5,4,6] [--flat 2.0]
       [--costs 0,7,14] [--winsor 25] [--start 2015-01-01]
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf  # noqa: E402

from backtest.engine import STOCK_UNIVERSE as _UNIVERSE  # noqa: E402
from data.sector_data import get_sector  # noqa: E402
from utils.symbols import to_yf_symbol  # noqa: E402


def _sweep(label: str, rows: list[tuple[float, int]], costs: list[float]) -> None:
    if not rows:
        print(f"  {label:28} (no events)")
        return
    vals = [r for r, _ in rows]
    n = len(vals)
    gross = statistics.mean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    by_year: dict[int, list[float]] = defaultdict(list)
    for v, yr in rows:
        by_year[yr].append(v)
    print(f"  {label:28} n={n:6}  gross={gross:+.3f}%")
    for c in costs:
        net = gross - c / 100.0
        t = net / se if se else 0.0
        pos = sum(1 for ys in by_year.values() if (statistics.mean(ys) - c / 100.0) > 0)
        star = " *" if abs(t) >= 2 and net > 0 else ""
        print(f"      @ {c:5.1f}bps  net={net:+.3f}%  t={t:+.2f}  +yrs={pos}/{len(by_year)}{star}")


def main() -> None:
    ap = argparse.ArgumentParser(description="peer_lead_lag backtest (v2 #5)")
    ap.add_argument("--hold", type=int, default=3)
    ap.add_argument("--peer-min", default="2.5,4,6", help="peer-move thresholds to sweep (%)")
    ap.add_argument("--flat", type=float, default=2.0)
    ap.add_argument("--costs", default="0,7,14")
    ap.add_argument("--winsor", type=float, default=25.0)
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()
    costs = [float(c) for c in args.costs.split(",")]
    peer_mins = [float(x) for x in args.peer_min.split(",")]
    base = min(peer_mins)
    w, h = args.winsor, args.hold

    universe = [to_yf_symbol(s) for s in _UNIVERSE]
    sectors = {to_yf_symbol(s): get_sector(s) for s in _UNIVERSE}
    syms = sorted(set(universe) | {"SPY"})
    px = yf.download(syms, start=args.start, auto_adjust=True, progress=False)["Close"].dropna(
        how="all"
    )
    spy = px["SPY"]
    cols = [c for c in px.columns if c != "SPY"]
    close = px[cols]
    print(f"prices: {len(cols)} names, {len(close)} sessions ({close.index[0].date()}+)")

    me5 = ((close / close.shift(5) - 1.0) * 100.0).sub((spy / spy.shift(5) - 1.0) * 100.0, axis=0)
    members: dict[str, list[str]] = defaultdict(list)
    for sym in cols:
        sec = sectors.get(sym) or "Unknown"
        if sec != "Unknown":
            members[sec].append(sym)
    members = {s: m for s, m in members.items() if len(m) >= 6}

    fwd = close.shift(-1 - h) / close.shift(-1) - 1.0
    spy_fwd = (spy.shift(-1 - h) / spy.shift(-1) - 1.0).reindex(close.index)
    fwd_excess = (fwd.sub(spy_fwd, axis=0) * 100.0).clip(-w, w)

    arms: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for i in range(25, len(close) - h - 2, 5):
        yr = close.index[i].year
        me_row, fx = me5.iloc[i], fwd_excess.iloc[i]
        for mem in members.values():
            vals = {s: me_row.get(s) for s in mem}
            vals = {s: float(v) for s, v in vals.items() if v is not None and not math.isnan(v)}
            if len(vals) < 6:
                continue
            tot = sum(vals.values())
            for sym, own in vals.items():
                peers = (tot - own) / (len(vals) - 1)  # ex-self equal-weight peer me5
                ex = fx.get(sym)
                if ex is None or (isinstance(ex, float) and math.isnan(ex)):
                    continue
                flat = -args.flat <= own <= args.flat
                if flat:
                    for pm in peer_mins:
                        if peers >= pm:
                            arms[f"laggard peers>={pm:.0f}%"].append((float(ex), yr))
                    if peers < base:
                        arms["flat, no peer signal (ctrl)"].append((float(ex), yr))
                elif peers >= base and own >= base:
                    arms["moved-with-peers (ctrl)"].append((float(ex), yr))

    print(f"\n=== peer_lead_lag — hold {h}d, t+1, winsorised excess vs SPY ===")
    labels = [f"laggard peers>={pm:.0f}%" for pm in peer_mins]
    labels += ["moved-with-peers (ctrl)", "flat, no peer signal (ctrl)"]
    for label in labels:
        _sweep(label, arms[label], costs)
    print(
        "\n  SHIPS if 'laggard' is net>0 @7bps with |t|>=2 AND beats both controls (the catch-up is "
        "real, not sector beta). Expected: arbitraged away in liquid names."
    )


if __name__ == "__main__":
    main()
