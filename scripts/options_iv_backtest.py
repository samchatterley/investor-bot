"""Options-signal kill/keep test — first-ever measurement of the live-only options signals.

Four options signals have run live since v1.98 with zero validation possible (no historical options
data). Alpaca's historical option bars (OPRA, ~Feb 2024→now, free with our key) now give ~2.4 years.
Contracts are addressed by SYNTHETIC OCC symbol (underlying + third-Friday expiry + strike near
price) — no chain discovery needed; verified working on expired mid-cap contracts.

Premises tested (weekly Friday snapshot, entry t+1 Monday, winsorised excess vs SPY, cost swept):
  iv_vs_rv_spread      ATM-call IV (Black-Scholes inversion of the close) / realised vol20 < 0.70
                       → long, hold 4 (the live signal, tested directly)
  put_call_contrarian  put/call VOLUME ratio > 2 at the ATM strike → long, hold 3
                       (live signal uses OI ratio 2.5 — volume is the available proxy; flagged)
  unusual_options_activity   ATM call-volume z-score vs own trailing 8 weeks >= 2 → long, hold 3
                       (live signal uses OTM call OI surge — volume proxy; flagged)
  options_skew_signal  NOT TESTABLE — needs OTM put chains (deferred; would need full-chain pulls)

CAVEATS: 2.4y window (corroborative, not 9y-definitive; the >=6/9-years bar cannot apply — read
sign + t + per-year consistency); European BS on American options + flat r=4.3% (small bias, fine
for thresholds); volume proxies where OI is unavailable.

Usage: python scripts/options_iv_backtest.py [--limit 907] [--start 2024-02-05] [--costs 0,7,14]
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import yfinance as yf  # noqa: E402

import config  # noqa: E402
from backtest.engine import STOCK_UNIVERSE as _UNIVERSE  # noqa: E402
from utils.symbols import to_yf_symbol  # noqa: E402

_R = 0.043  # flat risk-free (2024-2026 T-bill range); small IV-level bias, fine for thresholds


def _third_friday(y: int, m: int) -> date:
    d = date(y, m, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_call(s: float, k: float, t: float, sigma: float) -> float:
    if sigma <= 0 or t <= 0:
        return max(s - k, 0.0)
    d1 = (math.log(s / k) + (_R + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return s * _norm_cdf(d1) - k * math.exp(-_R * t) * _norm_cdf(d2)


def _implied_vol(price: float, s: float, k: float, t: float) -> float | None:
    """Bisection IV from a call price; None if outside no-arbitrage bounds."""
    intrinsic = max(s - k * math.exp(-_R * t), 0.0)
    if price <= intrinsic + 1e-4 or price >= s:
        return None
    lo, hi = 0.01, 4.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _bs_call(s, k, t, mid) < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _occ(sym: str, exp: date, cp: str, strike: float) -> str:
    return f"{sym}{exp.strftime('%y%m%d')}{cp}{int(round(strike * 1000)):08d}"


def _sweep(label: str, rows: list[tuple[float, int]], costs: list[float]) -> None:
    if not rows:
        print(f"  {label:30} (no events)")
        return
    vals = [r for r, _ in rows]
    n = len(vals)
    gross = statistics.mean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    by_year: dict[int, list[float]] = defaultdict(list)
    for v, yr in rows:
        by_year[yr].append(v)
    print(f"  {label:30} n={n:5}  gross={gross:+.3f}%")
    for c in costs:
        net = gross - c / 100.0
        t = net / se if se else 0.0
        pos = sum(1 for ys in by_year.values() if (statistics.mean(ys) - c / 100.0) > 0)
        star = " *" if abs(t) >= 2 and net > 0 else ""
        print(f"      @ {c:5.1f}bps  net={net:+.3f}%  t={t:+.2f}  +yrs={pos}/{len(by_year)}{star}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Options-signal kill/keep (Alpaca history)")
    ap.add_argument("--limit", type=int, default=len(_UNIVERSE))
    ap.add_argument("--start", default="2024-02-05")
    ap.add_argument("--costs", default="0,7,14")
    ap.add_argument("--winsor", type=float, default=25.0)
    args = ap.parse_args()
    costs = [float(c) for c in args.costs.split(",")]
    w = args.winsor

    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import OptionBarsRequest
    from alpaca.data.timeframe import TimeFrame

    oc = OptionHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)

    universe = [to_yf_symbol(s) for s in list(_UNIVERSE)[: args.limit]]
    syms = sorted(set(universe) | {"SPY"})
    px = yf.download(syms, start=args.start, auto_adjust=True, progress=False)["Close"].dropna(
        how="all"
    )
    spy = px["SPY"]
    cols = [c for c in px.columns if c != "SPY"]
    close = px[cols]
    logret = np.log(close / close.shift(1))
    rv20 = logret.rolling(20).std() * math.sqrt(252)
    print(f"prices: {len(cols)} names, {len(close)} sessions ({close.index[0].date()}+)")

    fridays = [i for i in range(21, len(close) - 6) if close.index[i].weekday() == 4]
    print(f"{len(fridays)} weekly snapshots; fetching option bars (batched) …", flush=True)

    # weekly records: (week_idx, sym, iv_rv, pc_vol_ratio, call_vol)
    recs: list[tuple[int, str, float | None, float | None, float]] = []
    for wnum, i in enumerate(fridays):
        d = close.index[i].date()
        want: dict[str, tuple[str, float, float, date]] = {}  # occ -> (sym, strike, spot, expiry)
        for sym in cols:
            p = close[sym].iloc[i]
            if not (2.0 < p < 500.0) or math.isnan(p):
                continue
            exp = _third_friday(d.year, d.month)
            if (exp - d).days < 10:
                nm = d.replace(day=1) + timedelta(days=32)
                exp = _third_friday(nm.year, nm.month)
            base = sym.replace("-", ".")  # OCC uses dots for class shares
            for k in {round(p), round(p / 2.5) * 2.5}:
                if k <= 0:
                    continue
                for cp in ("C", "P"):
                    want[_occ(base, exp, cp, k)] = (sym, k, p, exp)
        occ_syms = list(want)
        bars: dict[str, tuple[float, float]] = {}  # occ -> (close, volume)
        for j in range(0, len(occ_syms), 200):
            chunk = occ_syms[j : j + 200]
            try:
                got = oc.get_option_bars(
                    OptionBarsRequest(
                        symbol_or_symbols=chunk,
                        timeframe=TimeFrame.Day,
                        start=datetime(d.year, d.month, d.day),
                        end=datetime(d.year, d.month, d.day) + timedelta(days=1),
                    )
                ).data
            except Exception:
                continue
            for s_, blist in got.items():
                if blist:
                    bars[s_] = (float(blist[-1].close), float(blist[-1].volume))
        # assemble per-name: prefer the strike whose CALL traded
        per_name: dict[str, dict] = {}
        for occ_sym, (sym, k, _p, exp) in want.items():
            if occ_sym not in bars:
                continue
            cp = occ_sym[len(sym.replace("-", ".")) + 6]
            slot = per_name.setdefault(sym, {})
            key = (k, exp)
            slot.setdefault(key, {})[cp] = bars[occ_sym]
        for sym, strikes in per_name.items():
            best = None
            for (k, exp), sides in strikes.items():
                cvol = sides.get("C", (0.0, 0.0))[1]
                if "C" in sides and cvol > 0 and (best is None or cvol > best[3]):
                    best = (k, exp, sides, cvol)
            if best is None:
                continue
            k, exp, sides, _ = best
            p = close[sym].iloc[i]
            t_yr = max((exp - d).days, 1) / 365.0
            iv = _implied_vol(sides["C"][0], float(p), float(k), t_yr)
            rv = rv20[sym].iloc[i]
            iv_rv = (iv / rv) if (iv and rv and rv > 0.03) else None
            pvol = sides.get("P", (0.0, 0.0))[1]
            cvol = sides["C"][1]
            pc = (pvol / cvol) if cvol > 0 else None
            recs.append((wnum, sym, iv_rv, pc, cvol))
        if (wnum + 1) % 20 == 0:
            print(f"  ...week {wnum + 1}/{len(fridays)}, {len(recs)} name-weeks", flush=True)

    print(f"collected {len(recs)} name-week option snapshots")
    # forward returns per arm
    spy_s = spy
    call_hist: dict[str, list[float]] = defaultdict(list)
    arms: dict[str, list[tuple[float, int]]] = defaultdict(list)

    def _fx(i: int, sym: str, h: int) -> float | None:
        if i + 1 + h >= len(close):
            return None
        r = close[sym].iloc[i + 1 + h] / close[sym].iloc[i + 1] - 1.0
        sr = spy_s.iloc[i + 1 + h] / spy_s.iloc[i + 1] - 1.0
        if math.isnan(r) or math.isnan(sr):
            return None
        return max(-w, min(w, (r - sr) * 100.0))

    for wnum, sym, iv_rv, pc, cvol in recs:
        i = fridays[wnum]
        yr = close.index[i].year
        if iv_rv is not None:
            ex = _fx(i, sym, 4)
            if ex is not None:
                if iv_rv < 0.70:
                    arms["iv_vs_rv<0.70 (LIVE premise)"].append((ex, yr))
                if iv_rv > 1.30:
                    arms["iv_rv>1.30 (control: rich vol)"].append((ex, yr))
        if pc is not None:
            ex = _fx(i, sym, 3)
            if ex is not None and pc > 2.0:
                arms["put/call vol>2 (contrarian)"].append((ex, yr))
        hist = call_hist[sym]
        if len(hist) >= 8 and cvol > 0:
            mu = statistics.mean(hist[-8:])
            sd = statistics.stdev(hist[-8:]) if len(hist[-8:]) > 1 else 0.0
            if sd > 0 and (cvol - mu) / sd >= 2.0:
                ex = _fx(i, sym, 3)
                if ex is not None:
                    arms["call-vol z>=2 (unusual activity)"].append((ex, yr))
        hist.append(cvol)

    print("\n=== Options-signal premises — 2.4y Alpaca history, t+1 entry, excess vs SPY ===")
    for label in (
        "iv_vs_rv<0.70 (LIVE premise)",
        "iv_rv>1.30 (control: rich vol)",
        "put/call vol>2 (contrarian)",
        "call-vol z>=2 (unusual activity)",
    ):
        _sweep(label, arms[label], costs)
    print("  options_skew_signal: NOT TESTED (needs OTM put chains — deferred)")
    print(
        "\n  KEEP a live signal if its premise arm is net>0 with |t|>=2 and year-consistent; "
        "KILL if flat/negative. 2.4y window — corroborative, not definitive."
    )


if __name__ == "__main__":
    main()
