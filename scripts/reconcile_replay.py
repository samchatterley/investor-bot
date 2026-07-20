"""Replay-fidelity reconciliation runner (substrate brick 3).

For every logged live observation, reconstruct the same snapshot through the replay machinery
(point-in-time OHLCV slice -> technical features -> signals) and compare it to what the bot actually
recorded live. Persists a fidelity summary the weekly review renders. If replay cannot reproduce the
past, its counterfactuals are fiction -- so this is the gate counterfactual replay must clear.

Run:  python scripts/reconcile_replay.py
"""

from __future__ import annotations  # pragma: no cover

import os  # pragma: no cover
import sys  # pragma: no cover
from datetime import date, timedelta  # pragma: no cover

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # pragma: no cover

from backtest.replay import _build_preloaded  # noqa: E402  # pragma: no cover
from data.as_of import split_adjust_as_of  # noqa: E402  # pragma: no cover
from data.market_data import fetch_stock_data, summarise_for_ai  # noqa: E402  # pragma: no cover
from experiment.monitoring import load_scored_observations  # noqa: E402  # pragma: no cover
from experiment.reconciliation import (  # noqa: E402  # pragma: no cover
    save_reconciliation_summary,
    summarise,
)
from signals.evaluator import evaluate_signals  # noqa: E402  # pragma: no cover


def main() -> None:  # pragma: no cover
    obs = [o for o in load_scored_observations() if o.get("date") and o.get("symbol")]
    if not obs:
        print("No dated observations to reconcile.")
        return

    dates = [date.fromisoformat(o["date"]) for o in obs]
    symbols = sorted({o["symbol"] for o in obs})
    # Raw (unadjusted) prices + splits, so each snapshot is reconstructed with only the splits known as of
    # its own decision date -- point-in-time, unlike auto_adjust's retroactive (lookahead) factors.
    preloaded = _build_preloaded(
        symbols, min(dates) - timedelta(days=400), max(dates), unadjusted=True
    )

    pairs = []
    for o in obs:
        sym, as_of = o["symbol"], o["date"]
        if sym not in preloaded:
            continue
        as_of_frame = split_adjust_as_of(preloaded[sym], as_of)
        df = fetch_stock_data(sym, preloaded={sym: as_of_frame}, as_of=as_of)
        if df is None:
            continue
        sim = summarise_for_ai(sym, df, is_preloaded=True)
        sim["fired_signals"] = evaluate_signals(sim)
        live = {**(o.get("features") or {}), "fired_signals": o.get("fired_signals") or []}
        pairs.append((live, sim))

    summary = summarise(pairs)
    save_reconciliation_summary(summary)
    print(
        f"Reconciled {summary['n_pairs']} snapshot(s): replay fidelity {summary['fidelity']:.1%} "
        f"(worst-drifting field: {summary['worst_field']})."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
