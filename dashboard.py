"""
InvestorBot Web Dashboard

Launch:  python cli.py dashboard
         python -m streamlit run dashboard.py
         docker-compose up dashboard
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

import config
from utils.portfolio_tracker import load_history
from utils.decision_log import load_decisions

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="InvestorBot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("InvestorBot")
st.sidebar.caption("Autonomous AI Trading Bot")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Trades", "AI Decisions", "Backtest", "Diagnostics"],
)

mode_badge = "PAPER" if config.IS_PAPER else "LIVE"
halted = os.path.exists(config.HALT_FILE)
st.sidebar.divider()
st.sidebar.markdown(f"**Mode:** `{mode_badge}`")
st.sidebar.markdown(f"**Status:** {'🔴 HALTED' if halted else '🟢 Active'}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_account():
    try:
        from execution import trader
        client = trader.get_client()
        acc = trader.get_account_info(client)
        positions = trader.get_open_positions(client)
        return acc, positions
    except Exception as e:
        return None, []


def _load_latest_diagnostics() -> dict | None:
    """Load the most recent test report from logs/."""
    try:
        reports = sorted(
            [f for f in os.listdir(config.LOG_DIR) if f.startswith("test_report_") and f.endswith(".json")],
            reverse=True,
        )
        if not reports:
            return None
        with open(os.path.join(config.LOG_DIR, reports[0])) as f:
            return json.load(f)
    except Exception:
        return None


def _load_backtest() -> dict | None:
    path = os.path.join(config.LOG_DIR, "backtest_results.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ── Overview ──────────────────────────────────────────────────────────────────

if page == "Overview":
    st.title("Portfolio Overview")

    acc, positions = _load_account()

    if acc:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Portfolio Value", f"${acc['portfolio_value']:,.2f}")
        col2.metric("Cash", f"${acc['cash']:,.2f}")
        invested = acc["portfolio_value"] - acc["cash"]
        col3.metric("Invested", f"${invested:,.2f}")
        col4.metric("Open Positions", f"{len(positions)} / {config.MAX_POSITIONS}")
    else:
        st.warning("Could not connect to Alpaca — showing historical data only.")

    # P&L equity curve from daily history
    history = load_history()
    open_runs = [r for r in history if not r["date"].endswith(("-midday", "-close"))]
    if open_runs:
        st.subheader("Portfolio Equity Curve")
        curve_df = pd.DataFrame([
            {"Date": r["date"], "Value": r["account_after"]["portfolio_value"]}
            for r in open_runs
        ])
        curve_df["Date"] = pd.to_datetime(curve_df["Date"])
        curve_df = curve_df.set_index("Date")
        st.line_chart(curve_df["Value"])

        # Daily P&L bar chart
        st.subheader("Daily P&L")
        pnl_df = pd.DataFrame([
            {"Date": r["date"], "P&L": round(r.get("daily_pnl", 0), 2)}
            for r in open_runs[-30:]
        ])
        pnl_df["Date"] = pd.to_datetime(pnl_df["Date"])
        pnl_df = pnl_df.set_index("Date")
        st.bar_chart(pnl_df["P&L"])

    # Open positions table
    if positions:
        st.subheader("Open Positions")
        pos_df = pd.DataFrame([{
            "Symbol": p["symbol"],
            "Value ($)": round(p["market_value"], 2),
            "Unrealised P&L ($)": round(p["unrealized_pl"], 2),
            "Unrealised P&L (%)": round(p["unrealized_plpc"], 2),
        } for p in positions])
        st.dataframe(pos_df, use_container_width=True, hide_index=True)


# ── Trades ────────────────────────────────────────────────────────────────────

elif page == "Trades":
    st.title("Trade History")

    history = load_history()
    open_runs = [r for r in history if not r["date"].endswith(("-midday", "-close"))]

    if not open_runs:
        st.info("No trade history found yet.")
    else:
        rows = []
        for r in reversed(open_runs):
            for t in r.get("trades_executed", []):
                rows.append({
                    "Date": r["date"],
                    "Action": t.get("action", "?"),
                    "Symbol": t.get("symbol", "?"),
                    "Detail": t.get("detail", ""),
                    "Market": r.get("market_summary", "")[:60],
                })

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Summary stats
            st.subheader("Summary")
            col1, col2, col3 = st.columns(3)
            buys = sum(1 for r in rows if r["Action"] == "BUY")
            sells = sum(1 for r in rows if r["Action"] in ("SELL", "SELL_PARTIAL"))
            col1.metric("Total Buys", buys)
            col2.metric("Total Sells", sells)
            total_pnl = sum(r.get("daily_pnl", 0) for r in open_runs)
            col3.metric("Cumulative P&L", f"${total_pnl:+,.2f}")
        else:
            st.info("No trades executed yet.")


# ── AI Decisions ──────────────────────────────────────────────────────────────

elif page == "AI Decisions":
    st.title("AI Decision Log")
    st.caption("Every recommendation Claude made — whether executed or not")

    entries = load_decisions(n=500)
    if not entries:
        st.info("No AI decision records yet. They appear after the first trading run.")
    else:
        rows = []
        for e in reversed(entries):
            rows.append({
                "Date": e.get("date", ""),
                "Symbol": e.get("symbol", ""),
                "Action": e.get("action", ""),
                "Confidence": e.get("confidence", ""),
                "Executed": "Yes" if e.get("executed") else "No",
                "Signal": e.get("key_signal", ""),
                "Reasoning": e.get("reasoning", "")[:120],
                "Market": e.get("market_summary", "")[:60],
            })

        df = pd.DataFrame(rows)

        # Filters
        col1, col2 = st.columns(2)
        action_filter = col1.multiselect("Action", ["BUY", "SELL", "HOLD"], default=["BUY", "SELL", "HOLD"])
        exec_filter = col2.multiselect("Executed", ["Yes", "No"], default=["Yes", "No"])
        df = df[df["Action"].isin(action_filter) & df["Executed"].isin(exec_filter)]

        st.dataframe(df, use_container_width=True, hide_index=True)

        # Confidence distribution
        conf_vals = [e.get("confidence") for e in entries if e.get("confidence") is not None]
        if conf_vals:
            st.subheader("Confidence Score Distribution")
            conf_df = pd.DataFrame(conf_vals, columns=["Confidence"])
            st.bar_chart(conf_df["Confidence"].value_counts().sort_index())


# ── Backtest ──────────────────────────────────────────────────────────────────

elif page == "Backtest":
    st.title("Backtest Results")

    results = _load_backtest()
    if not results:
        st.info("No backtest results yet. Run: `python cli.py backtest --start 2024-01-01 --end 2024-12-31`")
    else:
        st.caption(f"Period: {results['start']} → {results['end']}  |  Initial capital: ${results['initial_capital']:,.2f}")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Return", f"{results['total_return_pct']:+.1f}%")
        col2.metric("Win Rate", f"{results['win_rate_pct']:.0f}%")
        col3.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
        col4.metric("Max Drawdown", f"{results['max_drawdown_pct']:.1f}%")
        col5.metric("Total Trades", results["total_trades"])

        # Equity curve
        if results.get("equity_curve"):
            st.subheader("Equity Curve")
            eq_df = pd.DataFrame(results["equity_curve"], columns=["Date", "Value"])
            eq_df["Date"] = pd.to_datetime(eq_df["Date"])
            eq_df = eq_df.set_index("Date")
            st.line_chart(eq_df["Value"])

        # Signal breakdown
        if results.get("by_signal"):
            st.subheader("Performance by Signal")
            sig_rows = []
            for sig, data in results["by_signal"].items():
                total = data["wins"] + data["losses"]
                wr = data["wins"] / total * 100 if total else 0
                avg = data["total_return"] / total if total else 0
                sig_rows.append({
                    "Signal": sig,
                    "Trades": total,
                    "Wins": data["wins"],
                    "Win Rate (%)": round(wr, 1),
                    "Avg Return (%)": round(avg, 2),
                })
            sig_df = pd.DataFrame(sig_rows)
            st.dataframe(sig_df, use_container_width=True, hide_index=True)

        # Trade log
        if results.get("trades"):
            with st.expander("Full trade log"):
                closed = [t for t in results["trades"] if t.get("action") == "SELL" and "pnl_pct" in t]
                if closed:
                    trade_df = pd.DataFrame(closed)
                    st.dataframe(trade_df, use_container_width=True, hide_index=True)


# ── Diagnostics ───────────────────────────────────────────────────────────────

elif page == "Diagnostics":
    st.title("System Diagnostics")

    report = _load_latest_diagnostics()
    if not report:
        st.info("No diagnostic report yet. Run: `python run_diagnostics.py`")
    else:
        ts = report.get("timestamp", "")
        st.caption(f"Last run: {ts}")

        status = report.get("status", "UNKNOWN")
        if status == "PASS":
            st.success(f"All tests passing — {report['passed']}/{report['total']} in {report['duration_seconds']}s")
        else:
            st.error(f"Tests failing — {report['passed']}/{report['total']} passed")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tests", report["total"])
        col2.metric("Passed", report["passed"])
        col3.metric("Failed", report.get("failed", 0))
        col4.metric("Errors", report.get("errors", 0))

        if report.get("failures"):
            st.subheader("Failures")
            for f in report["failures"]:
                with st.expander(f["test"]):
                    st.code(f["message"])

        # Run diagnostics button
        st.divider()
        if st.button("Run diagnostics now"):
            with st.spinner("Running tests..."):
                from run_diagnostics import run_diagnostics
                new_report = run_diagnostics()
                st.rerun()
