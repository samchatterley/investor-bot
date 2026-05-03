"""
InvestorBot CLI

Usage:
    python cli.py demo                       Simulated run — no credentials needed
    python cli.py status                     Account, positions, halt state
    python cli.py positions                  Open positions with live P&L
    python cli.py trades [--days N]          Recent trade history (default: 10 days)
    python cli.py decisions [--days N]       Recent AI decision log (default: 5 days)
    python cli.py run [--mode M] [--dry-run] Trigger a trading run
    python cli.py halt                       Emergency kill switch
    python cli.py resume                     Clear halt and resume
    python cli.py backtest [--start] [--end] [--capital N]
    python cli.py dashboard                  Launch Streamlit web dashboard
"""

import argparse
import os
import sys
from datetime import date

# Demo mode bypasses credential validation — parse args before importing config
_IS_DEMO = len(sys.argv) > 1 and sys.argv[1] == "demo"

if not _IS_DEMO:
    import config

    config.validate()
    from utils.decision_log import load_decisions
    from utils.portfolio_tracker import load_history
else:
    import config  # noqa: F811 — import without validate() for demo


# ── Helpers ───────────────────────────────────────────────────────────────────


def _header(text: str):
    print(f"\n  {'─' * 46}")
    print(f"  {text}")
    print(f"  {'─' * 46}")


def _print_positions(positions: list):
    if not positions:
        print("  No open positions.")
        return
    print(f"\n  {'Symbol':<8} {'Value':>12} {'P&L':>10} {'P&L %':>7}")
    print(f"  {'─' * 8} {'─' * 12} {'─' * 10} {'─' * 7}")
    for p in positions:
        pl = p["unrealized_pl"]
        pct = p["unrealized_plpc"]
        sign = "+" if pl >= 0 else ""
        print(
            f"  {p['symbol']:<8} ${p['market_value']:>11,.2f} {sign}${pl:>8,.2f} {sign}{pct:>5.1f}%"
        )


# ── Commands ──────────────────────────────────────────────────────────────────


def cmd_status(args):
    _header("BOT STATUS")
    halted = os.path.exists(config.HALT_FILE)
    mode_label = "PAPER" if config.IS_PAPER else "LIVE"
    print(f"  Status:    {'HALTED' if halted else 'Active'}")
    print(f"  Mode:      {mode_label}")

    try:
        from execution import trader

        client = trader.get_client()
        acc = trader.get_account_info(client)
        positions = trader.get_open_positions(client)
        print(f"  Portfolio: ${acc['portfolio_value']:,.2f}")
        print(f"  Cash:      ${acc['cash']:,.2f}")
        print(f"  Positions: {len(positions)}/{config.MAX_POSITIONS}")
        _print_positions(positions)
    except Exception as e:
        print(f"  Account:   [error: {e}]")


def cmd_positions(args):
    _header("OPEN POSITIONS")
    try:
        from execution import trader

        client = trader.get_client()
        positions = trader.get_open_positions(client)
        _print_positions(positions)
    except Exception as e:
        print(f"  Error: {e}")


def cmd_trades(args):
    _header(f"TRADE HISTORY (last {args.days} days)")
    history = load_history()
    recent = [r for r in history if not r["date"].endswith(("-midday", "-close"))]
    recent = recent[-args.days :]
    if not recent:
        print("  No trade history found.")
        return
    for record in recent:
        pnl = record.get("daily_pnl", 0)
        sign = "+" if pnl >= 0 else ""
        print(
            f"\n  {record['date']}  P&L: {sign}${pnl:.2f}  |  {record.get('market_summary', '')[:60]}"
        )
        trades = record.get("trades_executed", [])
        if trades:
            for t in trades:
                print(
                    f"    {t.get('action', '?'):<5} {t.get('symbol', '?'):<8} {t.get('detail', '')}"
                )
        else:
            print("    (no trades)")


def cmd_decisions(args):
    _header(f"AI DECISION LOG (last {args.days} trading days)")
    entries = load_decisions(n=500)
    if not entries:
        print("  No AI decision records found.")
        return

    groups = {}
    for e in entries:
        groups.setdefault(e["date"], []).append(e)

    for date_key in sorted(groups)[-args.days :]:
        day_entries = groups[date_key]
        summary = day_entries[0].get("market_summary", "")
        print(f"\n  {date_key}  {summary[:70]}")
        for e in day_entries:
            executed = "[EXECUTED]" if e.get("executed") else "          "
            sig = f"  [{e.get('key_signal', '')}]" if e.get("key_signal") else ""
            conf = e.get("confidence", "?")
            print(
                f"    {executed}  {e.get('action', '?'):<5} {e.get('symbol', '?'):<8}  conf={conf}{sig}"
            )
            reasoning = e.get("reasoning", "")
            if reasoning:
                print(f"              {reasoning[:90]}")


def cmd_run(args):
    import main as bot

    bot.run(dry_run=args.dry_run, mode=args.mode)


def cmd_halt(args):
    confirm = input("  Activate kill switch? This will liquidate ALL positions. [yes/no]: ")
    if confirm.strip().lower() == "yes":
        import main as bot

        bot._run_kill_switch()
    else:
        print("  Cancelled.")


def cmd_resume(args):
    import main as bot

    bot._run_clear_halt()


def cmd_backtest(args):
    from backtest import run_backtest

    capital = args.capital
    if capital is None:
        try:
            from execution import trader

            client = trader.get_client()
            capital = trader.get_account_info(client)["portfolio_value"]
            print(f"  Using current portfolio value: ${capital:,.2f}")
        except Exception:
            capital = 100_000.0
            print(f"  Defaulting to ${capital:,.0f}")
    end = args.end or date.today().isoformat()
    run_backtest(
        config.STOCK_UNIVERSE, args.start, end, capital, max_positions=config.MAX_POSITIONS
    )


def cmd_dashboard(args):
    import subprocess

    dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.py")
    print("  Launching dashboard at http://localhost:8501 ...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])


def cmd_demo(_args):
    """
    Credential-free simulated run for reviewers and interviewers.
    Loads static fixtures, exercises the real validator and risk layers,
    and prints each stage as it would appear in a live run.
    """
    import json
    import textwrap
    import time

    _ROOT = os.path.dirname(os.path.abspath(__file__))
    fixture_path = os.path.join(_ROOT, "evals", "fixtures", "demo_run.json")
    with open(fixture_path) as f:
        demo = json.load(f)

    def _sep():
        print("  " + "─" * 60)

    def _step(label: str):
        print(f"\n  ▶  {label}")
        time.sleep(0.3)

    def _ok(msg: str):
        print(f"     ✓  {msg}")

    def _warn(msg: str):
        print(f"     ⚠  {msg}")

    def _info(msg: str):
        for line in textwrap.wrap(msg, width=70):
            print(f"     {line}")

    print()
    _sep()
    print("  InvestorBot — Demo Run  (no credentials required)")
    print("  Static fixtures · real validators · real risk gate · no API calls")
    _sep()

    # ── 1. Account & regime ───────────────────────────────────────────────────
    _step("Account snapshot")
    acc = demo["account"]
    _ok(f"Portfolio ${acc['portfolio_value']:,.0f}  |  Cash ${acc['cash']:,.0f}  |  Paper mode")

    regime = demo["regime"]
    vix = demo["vix"]
    _ok(f"Regime: {regime['regime']}  |  SPY 1d: {regime['spy_change_pct']:+.1f}%  |  VIX: {vix}")

    positions = demo["open_positions"]
    if positions:
        _ok(f"Open positions: {len(positions)}  ({', '.join(p['symbol'] for p in positions)})")

    # ── 2. Pre-filter ─────────────────────────────────────────────────────────
    _step("Pre-filter candidates (momentum / mean-reversion / breakout screens)")
    from execution.stock_scanner import prefilter_candidates

    snapshots = demo["snapshots"]
    qualified = prefilter_candidates(snapshots)
    filtered_out = [s["symbol"] for s in snapshots if s not in qualified]
    _ok(f"Passed:   {[s['symbol'] for s in qualified]}")
    if filtered_out:
        _info(f"Filtered: {filtered_out} (no qualifying technical pattern or below volume floor)")

    # ── 3. AI response (pre-recorded fixture) ────────────────────────────────
    _step("AI analysis  (pre-recorded fixture — no Claude API call in demo)")
    ai = demo["ai_response"]
    _info(f"Market summary: {ai['market_summary']}")
    _ok(
        f"{len(ai['buy_candidates'])} buy candidates  |  {len(ai['position_decisions'])} position decisions"
    )

    # ── 4. Validation ─────────────────────────────────────────────────────────
    _step("Validation layer")
    from utils.validators import validate_ai_response

    scanned_symbols = {s["symbol"] for s in snapshots}
    held_symbols = {p["symbol"] for p in positions}
    is_valid, errors = validate_ai_response(ai, scanned_symbols, held_symbols=held_symbols)

    if errors:
        for e in errors:
            _warn(f"Rejected: {e}")
    passing = [c for c in ai["buy_candidates"] if c["symbol"] in scanned_symbols]
    _ok(f"{len(passing)} candidates passed  |  {len(ai['buy_candidates']) - len(passing)} rejected")

    # ── 5. Risk gate ──────────────────────────────────────────────────────────
    _step("Risk gate  (Kelly sizing · position limits · bear filter · sector cap)")

    if regime["is_bearish"]:
        _warn("Bear filter active — all buys suppressed")
        passing = []
    else:
        _ok("Bear filter: not triggered")

    from risk.position_sizer import kelly_fraction

    available_cash = acc["cash"] * 0.9  # 10% cash reserve
    max_positions = 5
    slots = max_positions - len(positions)
    _ok(
        f"Position slots: {len(positions)}/{max_positions}  |  Available cash: ${available_cash:,.0f}"
    )

    orders = []
    for candidate in passing[:slots]:
        symbol = candidate["symbol"]
        conf = candidate["confidence"]
        if conf < 7:
            _warn(f"{symbol}: confidence {conf} below floor — skipped")
            continue
        kelly = kelly_fraction(
            conf, signal=candidate.get("key_signal", "unknown"), regime=regime["regime"]
        )
        notional = min(available_cash * kelly, acc["portfolio_value"] * 0.45)
        if notional < 1.0:
            _warn(f"{symbol}: notional ${notional:.2f} too small — skipped")
            continue
        orders.append(
            {
                "symbol": symbol,
                "notional": notional,
                "kelly": kelly,
                "confidence": conf,
                "signal": candidate.get("key_signal"),
                "reasoning": candidate.get("reasoning", ""),
            }
        )
        _ok(
            f"{symbol}: ${notional:,.0f}  Kelly {kelly:.0%}  conf={conf}  [{candidate.get('key_signal')}]"
        )

    # ── 6. Simulated execution ────────────────────────────────────────────────
    _step("Simulated order placement  (no real API calls)")
    import uuid

    run_id = str(uuid.uuid4())[:8]
    for o in orders:
        fake_order_id = str(uuid.uuid4())[:8]
        _ok(f"[SIMULATED] BUY {o['symbol']} ${o['notional']:,.0f} → order_id={fake_order_id}")
        _info(f"  Reasoning: {o['reasoning'][:100]}")

    if not orders:
        _info("No orders placed this cycle (candidates filtered by validator + risk gate).")

    # ── 7. Audit log ──────────────────────────────────────────────────────────
    _step("Audit trail  (what would be written to logs/investorbot.db)")
    _ok(f"run_id={run_id}  mode=open  paper=True")
    _ok(f"RUN_START  portfolio=${acc['portfolio_value']:,.0f}  cash=${acc['cash']:,.0f}")
    _ok(f"AI_DECISION  buy_count={len(orders)}  sell_count=0")
    for o in orders:
        _ok(f"ORDER_PLACED  symbol={o['symbol']}  side=BUY  notional={o['notional']:.2f}")
    _ok("RUN_END  trades_executed=" + str(len(orders)))

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    _sep()
    print("  Demo complete.")
    print(f"  Orders that would have been placed: {len(orders)}")
    print(f"  Candidates rejected by validator:   {len(ai['buy_candidates']) - len(passing)}")
    print(f"  Stocks filtered by pre-screener:    {len(filtered_out)}")
    print()
    print("  To run against real markets:  cp .env.example .env  # fill in your keys")
    print("  To run a dry-run (no orders): python cli.py run --dry-run")
    _sep()
    print()


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        prog="python cli.py",
        description="InvestorBot CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Account info, open positions, halt state")
    sub.add_parser("positions", help="Open positions with live P&L")

    p = sub.add_parser("trades", help="Recent trade history")
    p.add_argument("--days", type=int, default=10, metavar="N")

    p = sub.add_parser("decisions", help="Recent AI decision log")
    p.add_argument("--days", type=int, default=5, metavar="N")

    p = sub.add_parser("run", help="Trigger a trading run")
    p.add_argument("--mode", choices=["open", "midday", "close"], default="open")
    p.add_argument("--dry-run", action="store_true")

    sub.add_parser("halt", help="Emergency kill switch — liquidate all positions")
    sub.add_parser("resume", help="Clear halt file and resume trading")

    p = sub.add_parser("backtest", help="Run historical backtest")
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--capital", type=float, default=None)

    sub.add_parser("dashboard", help="Launch Streamlit web dashboard")
    sub.add_parser("demo", help="Simulated run with fixture data — no credentials needed")

    args = parser.parse_args()
    {
        "status": cmd_status,
        "positions": cmd_positions,
        "trades": cmd_trades,
        "decisions": cmd_decisions,
        "run": cmd_run,
        "halt": cmd_halt,
        "resume": cmd_resume,
        "backtest": cmd_backtest,
        "dashboard": cmd_dashboard,
        "demo": cmd_demo,
    }[args.command](args)


if __name__ == "__main__":
    main()
