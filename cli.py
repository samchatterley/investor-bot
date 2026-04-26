"""
InvestorBot CLI

Usage:
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
from itertools import groupby

import config
from utils.portfolio_tracker import load_history
from utils.decision_log import load_decisions


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
    print(f"  {'─'*8} {'─'*12} {'─'*10} {'─'*7}")
    for p in positions:
        pl = p["unrealized_pl"]
        pct = p["unrealized_plpc"]
        sign = "+" if pl >= 0 else ""
        print(f"  {p['symbol']:<8} ${p['market_value']:>11,.2f} {sign}${pl:>8,.2f} {sign}{pct:>5.1f}%")


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
    recent = recent[-args.days:]
    if not recent:
        print("  No trade history found.")
        return
    for record in recent:
        pnl = record.get("daily_pnl", 0)
        sign = "+" if pnl >= 0 else ""
        print(f"\n  {record['date']}  P&L: {sign}${pnl:.2f}  |  {record.get('market_summary','')[:60]}")
        trades = record.get("trades_executed", [])
        if trades:
            for t in trades:
                print(f"    {t.get('action','?'):<5} {t.get('symbol','?'):<8} {t.get('detail','')}")
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

    for date in sorted(groups)[-args.days:]:
        day_entries = groups[date]
        summary = day_entries[0].get("market_summary", "")
        print(f"\n  {date}  {summary[:70]}")
        for e in day_entries:
            executed = "[EXECUTED]" if e.get("executed") else "          "
            sig = f"  [{e.get('key_signal','')}]" if e.get("key_signal") else ""
            conf = e.get("confidence", "?")
            print(f"    {executed}  {e.get('action','?'):<5} {e.get('symbol','?'):<8}  conf={conf}{sig}")
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
    run_backtest(config.STOCK_UNIVERSE, args.start, args.end, capital)


def cmd_dashboard(args):
    import subprocess
    dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.py")
    print("  Launching dashboard at http://localhost:8501 ...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])


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
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--capital", type=float, default=None)

    sub.add_parser("dashboard", help="Launch Streamlit web dashboard")

    args = parser.parse_args()
    {
        "status":    cmd_status,
        "positions": cmd_positions,
        "trades":    cmd_trades,
        "decisions": cmd_decisions,
        "run":       cmd_run,
        "halt":      cmd_halt,
        "resume":    cmd_resume,
        "backtest":  cmd_backtest,
        "dashboard": cmd_dashboard,
    }[args.command](args)


if __name__ == "__main__":
    main()
