"""Runner for the forward-outcome backfill.

Reads the decision-time observations (logs/experiment_observations.jsonl), fetches the daily close
history for every symbol up to today, scores each observation's forward returns for any horizon that
has now closed (experiment/backfill.py — strictly point-in-time), and writes the scored dataset to
logs/experiment_scored.jsonl.

Re-runnable: it re-derives the whole scored file each run, so horizons fill in as they close over
time. Intended to run daily after the close (or on demand). Operational glue around live price data,
so it is excluded from unit-test coverage; the scoring logic lives in experiment/backfill.py.

Run:  python scripts/backfill_outcomes.py
"""

from __future__ import annotations  # pragma: no cover

import json  # pragma: no cover
import os  # pragma: no cover
import sys  # pragma: no cover
from datetime import date  # pragma: no cover

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # pragma: no cover

from experiment.backfill import DEFAULT_HORIZONS, backfill  # noqa: E402  # pragma: no cover

_OBS_PATH = os.path.join("logs", "experiment_observations.jsonl")  # pragma: no cover
_OUT_PATH = os.path.join("logs", "experiment_scored.jsonl")  # pragma: no cover


def _load_observations(path: str) -> list[dict]:  # pragma: no cover
    if not os.path.exists(path):
        return []
    rows: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _price_series(
    symbols: set[str], earliest: str
) -> dict[str, tuple[list[str], list[float]]]:  # pragma: no cover
    from data.market_data import _download_symbols

    fetch_days = (date.today() - date.fromisoformat(earliest)).days + 40
    data = _download_symbols(list(symbols), max(fetch_days, 60))
    series: dict[str, tuple[list[str], list[float]]] = {}
    for sym, df in data.items():
        dates = [ts.strftime("%Y-%m-%d") for ts in df.index]
        closes = [float(c) for c in df["Close"].tolist()]
        series[sym] = (dates, closes)
    return series


def main() -> None:  # pragma: no cover
    obs = _load_observations(_OBS_PATH)
    if not obs:
        print("No observations yet — nothing to backfill.")
        return

    symbols = {o["symbol"] for o in obs if o.get("symbol")}
    earliest = min(o["date"] for o in obs if o.get("date"))
    series = _price_series(symbols, earliest)
    scored = backfill(obs, series)

    os.makedirs(os.path.dirname(_OUT_PATH) or ".", exist_ok=True)
    with open(_OUT_PATH, "w") as fh:
        for row in scored:
            fh.write(json.dumps(row) + "\n")

    n_primary = sum(1 for r in scored if r["outcomes"].get("forward_r_5d") is not None)
    print(
        f"Scored {len(scored)}/{len(obs)} observations; {n_primary} with the 5d primary horizon closed."
    )
    for h in DEFAULT_HORIZONS:
        n = sum(1 for r in scored if r["outcomes"].get(f"forward_r_{h}d") is not None)
        print(f"  {h}d horizon closed: {n}")


if __name__ == "__main__":  # pragma: no cover
    main()
