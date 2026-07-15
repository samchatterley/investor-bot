"""Macro-gate efficacy report.

(1) Backfills past macro-skip events from the scheduler log + daily run records into
    logs/macro_gate_shadow.jsonl (idempotent — dedup by date+mode), so history is captured without
    needing the live capture to have been running.
(2) Scores each event's would-be buys forward (equal-weight market-excess vs SPY) at several horizons
    and prints, per event, whether the macro gate SAVED us (blocked names lagged the market, excess<0)
    or COST us (they beat it, excess>0).

Horizons: 0d = the skip-day itself (open->close, the immediate read); 1d/3d/5d = skip-day close ->
N-trading-days-later close (the forward drift the gate is really trying to avoid).

Operational glue around live prices + logs, so excluded from unit-test coverage; the scoring core lives
in analysis/macro_gate_shadow.py.

Run:  python scripts/macro_gate_report.py
"""

from __future__ import annotations  # pragma: no cover

import glob  # pragma: no cover
import json  # pragma: no cover
import os  # pragma: no cover
import re  # pragma: no cover
import sys  # pragma: no cover
from datetime import date  # pragma: no cover

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # pragma: no cover

from analysis.macro_gate_shadow import (  # noqa: E402  # pragma: no cover
    capture,
    load,
    score_event,
)

_LOG = os.path.join("logs", "scheduler.log")  # pragma: no cover
_HEADER = re.compile(r"Trading bot \| (\d{4}-\d{2}-\d{2}) \| mode=(\w+)")  # pragma: no cover
_SKIP = re.compile(r"Skipping new buys:.*macro event: ?([^,]*)")  # pragma: no cover
_HORIZONS = (0, 1, 3, 5)  # pragma: no cover


def _daily_json(d: str, mode: str) -> dict | None:  # pragma: no cover
    for path in glob.glob(os.path.join("logs", "*", "*", f"{d}-{mode}.json")):
        try:
            with open(path) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None
    return None


def _backfill() -> int:  # pragma: no cover
    """Scan scheduler.log for macro-skip runs; append any (date, mode) not already logged."""
    existing = {(r.get("date"), r.get("mode")) for r in load()}
    try:
        with open(_LOG) as fh:
            lines = fh.readlines()
    except OSError:
        return 0
    cur_date = cur_mode = None
    added = 0
    for line in lines:
        h = _HEADER.search(line)
        if h:
            cur_date, cur_mode = h.group(1), h.group(2)
            continue
        s = _SKIP.search(line)
        if s and cur_date and (cur_date, cur_mode) not in existing:
            j = _daily_json(cur_date, cur_mode) or {}
            capture(
                (s.group(1) or "").strip() or "(unspecified)",
                j.get("buy_candidates", []),
                today=cur_date,
                mode=cur_mode,
            )
            existing.add((cur_date, cur_mode))
            added += 1
    return added


def _series(symbols: list[str], earliest: str) -> dict:  # pragma: no cover
    from data.market_data import _download_symbols

    fetch_days = (date.today() - date.fromisoformat(earliest)).days + 12
    out: dict[str, tuple[list[str], list[float], list[float]]] = {}
    for sym, df in _download_symbols(symbols, max(fetch_days, 20)).items():
        dates = [ts.strftime("%Y-%m-%d") for ts in df.index]
        out[sym] = (dates, [float(o) for o in df["Open"]], [float(c) for c in df["Close"]])
    return out


def _px(
    series: dict, sym: str, d: str, field: str, fwd: int = 0
) -> float | None:  # pragma: no cover
    s = series.get(sym)
    if not s or d not in s[0]:
        return None
    i = s[0].index(d) + fwd
    if i < 0 or i >= len(s[0]):
        return None
    return (s[1] if field == "open" else s[2])[i]


def _prices_for(series: dict, syms: list[str], d: str, h: int) -> dict:  # pragma: no cover
    """(entry, exit) per symbol for horizon h. h=0 is intraday (open->close same day); h>0 is
    skip-day close -> +h-trading-day close."""
    if h == 0:
        return {s: (_px(series, s, d, "open"), _px(series, s, d, "close")) for s in syms}
    return {s: (_px(series, s, d, "close"), _px(series, s, d, "close", fwd=h)) for s in syms}


def main() -> None:  # pragma: no cover
    added = _backfill()
    events = sorted(load(), key=lambda e: (e["date"], e.get("mode") or ""))
    if not events:
        print("No macro-gate events logged yet.")
        return
    print(f"Backfilled {added} new event(s); {len(events)} total in the log.\n")

    syms = sorted({c["symbol"] for e in events for c in e["candidates"]} | {"SPY"})
    series = _series(syms, min(e["date"] for e in events))

    hdr = f"{'date':<11}{'mode':<11}{'event':<26}{'n':>3}  " + "  ".join(
        f"{h}d".rjust(8) for h in _HORIZONS
    )
    print(hdr)
    print("-" * len(hdr))
    agg: dict[int, list[float]] = {h: [] for h in _HORIZONS}
    for e in events:
        cells = []
        n_scored = 0
        for h in _HORIZONS:
            n, mean = score_event(
                e["candidates"],
                {
                    **_prices_for(series, syms, e["date"], h),
                    "SPY": _prices_for(series, ["SPY"], e["date"], h)["SPY"],
                },
            )
            n_scored = max(n_scored, n)
            if mean is None:
                cells.append("  --  ")
            else:
                cells.append(f"{mean:+7.2f}%")
                agg[h].append(mean)
        ev = (e.get("macro_event") or "")[:24]
        print(
            f"{e['date']:<11}{(e.get('mode') or ''):<11}{ev:<26}{n_scored:>3}  "
            + "  ".join(c.rjust(8) for c in cells)
        )

    print("-" * len(hdr))
    summ = []
    for h in _HORIZONS:
        xs = agg[h]
        summ.append(f"{(sum(xs) / len(xs)):+7.2f}%" if xs else "  --  ")
    print(f"{'MEAN excess (across events)':<51}     " + "  ".join(c.rjust(8) for c in summ))
    print(
        "\nReading: excess < 0 ⇒ blocked names lagged the market ⇒ the gate SAVED us; > 0 ⇒ it COST us."
    )
    print(
        "Per-event signs are anecdotes; the gate's value is variance reduction across many events."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
