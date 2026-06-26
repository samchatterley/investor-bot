"""Collect off-universe short "misses" — names the AI proposes to short that aren't in the scanned
short universe — WITHOUT placing trades or polluting the experiment dataset.

Reuses the live pipeline through the AI call + validation only: it does NOT call `_run_ai_phase`
(so no `log_run_observations` / decision-log writes), `_execute_*` (no orders), or `_finalise`
(no daily-save / email). For each rejected short it records the symbol + the catalyst flags that
name carries on the long-side snapshot, so "should we expand the short universe?" is data-driven.
Appends JSONL to logs/short_misses.jsonl.

It DOES make one real Anthropic call per mode and reads live broker/market data (both read-only /
cost-only). Safe to run during market hours — it never trades and never restarts anything.

Usage: python scripts/collect_short_misses.py --mode open    (and again --mode midday)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import UTC, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

import main  # noqa: E402
from config import LOG_DIR, today_et  # noqa: E402
from core.deps import TradingDeps  # noqa: E402
from utils.validators import validate_ai_response  # noqa: E402

_OUT = os.path.join(LOG_DIR, "short_misses.jsonl")
_MISS_RE = re.compile(r"SHORT candidate '([A-Z.\-]+)' not in scanned short universe")
# Long-side catalyst flags worth knowing for a missed short — tells us whether the AI was reaching
# for a name that actually had a shortable catalyst we'd surface if it were in the short universe.
_CATALYST_KEYS = (
    "guidance_negative",
    "secondary_offering",
    "accounting_concern",
    "regulatory_event",
    "insider_cluster",
    "analyst_downgrade",
    "eps_estimate_cut",
    "high_short_interest",
)


def collect(mode: str) -> list[dict]:
    deps = TradingDeps.production()
    client = deps.trader.get_client()
    snap = main._get_position_snapshot(client, deps=deps)
    mc = main._fetch_market_context(deps=deps)
    db = main._build_data_bundle(client, snap, mc, mode, deps=deps)
    if db is None:
        print("No data bundle (no snapshots) — nothing to collect.")
        return []

    account = deps.trader.get_account_info(client)
    decisions = deps.ai_analyst.get_trading_decisions(
        snapshots=db.ai_snapshots,
        current_positions=snap.open_positions,
        available_cash=account["cash"],
        portfolio_value=account["portfolio_value"],
        news_by_symbol=db.news,
        track_record=deps.portfolio_tracker.get_track_record(10),
        market_regime=mc.regime,
        position_ages=snap.position_ages,
        stale_positions=snap.stale,
        vix=mc.vix,
        sector_performance=mc.sector_perf,
        sentiment=db.sentiment,
        earnings_risk={sym: str(ed) for sym, ed in snap.earnings_risk.items()},
        macro_risk=mc.macro,
        leading_sectors=mc.leading_sectors,
        options_signals=db.options_sigs,
        lessons=mc.lessons,
        short_candidates=db.short_candidates,
        run_id="collect_short_misses",
    )
    if not decisions:
        print("AI returned no decisions.")
        return []

    known_short = {c["symbol"] for c in db.short_candidates}
    _, errors = validate_ai_response(
        decisions,
        {s["symbol"] for s in db.ai_snapshots},
        held_symbols=snap.held_symbols,
        known_short_symbols=known_short,
    )

    long_by_sym = {s["symbol"]: s for s in db.snapshots}
    offered = sorted(known_short)
    requested = [c.get("symbol") for c in decisions.get("short_candidates", [])]
    print(
        f"{mode}: regime={mc.regime.get('regime')} | offered {len(offered)} short candidates | "
        f"AI requested {requested or 'none'}"
    )

    misses: list[dict] = []
    for e in errors:
        m = _MISS_RE.search(e)
        if not m:
            continue
        sym = m.group(1)
        ls = long_by_sym.get(sym, {})
        misses.append(
            {
                "ts": datetime.now(UTC).isoformat(),
                "date": today_et().isoformat(),
                "mode": mode,
                "regime": mc.regime.get("regime"),
                "symbol": sym,
                "in_long_universe": sym in long_by_sym,
                "long_catalyst_flags": {k: bool(ls.get(k)) for k in _CATALYST_KEYS},
            }
        )
    return misses


def main_cli() -> None:
    ap = argparse.ArgumentParser(description="Collect off-universe short misses (read-only)")
    ap.add_argument("--mode", default="open", choices=["open", "open_sells", "midday", "close"])
    args = ap.parse_args()

    misses = collect(args.mode)
    print(f"\n{args.mode}: {len(misses)} off-universe short request(s)")
    for mm in misses:
        flags = [k for k, v in mm["long_catalyst_flags"].items() if v]
        print(
            f"  {mm['symbol']} | in_long_universe={mm['in_long_universe']} | "
            f"long catalyst flags: {flags or 'none'}"
        )
    if misses:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(_OUT, "a") as f:
            for mm in misses:
                f.write(json.dumps(mm) + "\n")
        print(f"  appended {len(misses)} record(s) → {_OUT}")


if __name__ == "__main__":
    main_cli()
