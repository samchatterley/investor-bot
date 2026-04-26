"""
InvestorBot Web Dashboard

Launch:  python cli.py dashboard
         streamlit run dashboard.py
"""

import json
import os

import pandas as pd
import plotly.graph_objects as go
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

# ── Design tokens ─────────────────────────────────────────────────────────────

C_BG      = "#0b0f1a"
C_SURFACE = "#131929"
C_BORDER  = "rgba(255,255,255,0.08)"
C_ACCENT  = "#00d4aa"
C_GREEN   = "#26c281"
C_RED     = "#e05c5c"
C_YELLOW  = "#f0b429"
C_TEXT    = "#e2e8f0"
C_MUTED   = "#8896a5"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=C_TEXT, family="system-ui, -apple-system, sans-serif", size=12),
    margin=dict(l=0, r=0, t=28, b=0),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", showline=False, zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", showline=False, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
)

# ── Global CSS — targets native Streamlit elements ────────────────────────────

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}}

/* Hide Streamlit chrome */
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {C_SURFACE} !important;
    border-right: 1px solid {C_BORDER};
}}
[data-testid="stSidebar"] h3 {{
    color: {C_ACCENT} !important;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}}

/* ── Metric cards — target native st.metric ── */
[data-testid="metric-container"] {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 12px;
    padding: 18px 20px 14px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
}}
[data-testid="metric-container"]:hover {{
    border-color: rgba(255,255,255,0.15);
    transition: border-color 0.2s;
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: {C_MUTED} !important;
}}
[data-testid="stMetricValue"] {{
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: {C_TEXT} !important;
}}
[data-testid="stMetricDelta"] svg {{ display: none; }}
[data-testid="stMetricDelta"] > div {{
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}}

/* ── Plotly chart containers ── */
.stPlotlyChart {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 12px;
    padding: 8px;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 10px;
    overflow: hidden;
}}

/* ── Buttons ── */
.stButton > button {{
    background: {C_ACCENT} !important;
    color: #0b0f1a !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.4rem !important;
}}
.stButton > button:hover {{
    background: #00b894 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,212,170,0.3) !important;
    transition: all 0.15s;
}}

/* ── Expander ── */
[data-testid="stExpander"] {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER} !important;
    border-radius: 10px !important;
}}

/* ── Section dividers ── */
hr {{ border-color: {C_BORDER} !important; }}

/* ── Info / warning boxes ── */
.stAlert {{
    background: rgba(0,212,170,0.08) !important;
    border: 1px solid rgba(0,212,170,0.2) !important;
    border-radius: 10px !important;
    color: {C_TEXT} !important;
}}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _section(title: str):
    st.markdown(f"<br>", unsafe_allow_html=True)
    st.markdown(f"**{title.upper()}**")
    st.markdown(
        f'<hr style="margin-top:2px;margin-bottom:12px;border-color:{C_BORDER};">',
        unsafe_allow_html=True,
    )


def _fmt_pct(v: float) -> str:
    return f"+{v:.2f}%" if v >= 0 else f"{v:.2f}%"


def _fmt_usd(v: float) -> str:
    return f"+${v:,.2f}" if v >= 0 else f"-${abs(v):,.2f}"


def _equity_fig(dates, values, color=C_ACCENT):
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=values,
        mode="lines",
        line=dict(color=color, width=2.5),
        fill="tozeroy",
        fillcolor=f"rgba({r},{g},{b},0.07)",
        hovertemplate="<b>%{x}</b><br>$%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def _load_account():
    try:
        from execution import trader
        client = trader.get_client()
        acc = trader.get_account_info(client)
        positions = trader.get_open_positions(client)
        return acc, positions, None
    except Exception as e:
        return None, [], str(e)


def _load_diagnostics() -> dict | None:
    try:
        reports = sorted(
            [f for f in os.listdir(config.LOG_DIR)
             if f.startswith("test_report_") and f.endswith(".json")],
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


# ── Sidebar ───────────────────────────────────────────────────────────────────

halted = os.path.exists(config.HALT_FILE)

with st.sidebar:
    st.markdown("### 📈 InvestorBot")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Overview", "Trades", "AI Decisions", "Backtest", "Diagnostics"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(f"**Status** &nbsp; {'🔴 HALTED' if halted else '🟢 Active'}")
    st.markdown(f"**Mode** &nbsp;&nbsp;&nbsp; {'🔴 LIVE' if not config.IS_PAPER else '🔵 PAPER'}")


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "Overview":
    st.markdown("## Portfolio Overview")

    acc, positions, err = _load_account()
    history = load_history()
    open_runs = [r for r in history if not r["date"].endswith(("-midday", "-close"))]

    if acc:
        pnl_today = open_runs[-1].get("daily_pnl", 0) if open_runs else 0
        invested  = acc["portfolio_value"] - acc["cash"]
        invested_pct = invested / acc["portfolio_value"] * 100 if acc["portfolio_value"] else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Portfolio Value", f"${acc['portfolio_value']:,.2f}", _fmt_usd(pnl_today))
        c2.metric("Cash", f"${acc['cash']:,.2f}", f"{100 - invested_pct:.0f}% idle")
        c3.metric("Invested", f"${invested:,.2f}", f"{invested_pct:.0f}% deployed")
        c4.metric("Open Positions", f"{len(positions)} / {config.MAX_POSITIONS}")
    elif err:
        st.warning(f"Could not connect to Alpaca ({err}) — showing historical data only.")

    if open_runs:
        _section("Equity Curve")
        df = pd.DataFrame([{
            "Date": pd.to_datetime(r["date"]),
            "Value": r["account_after"]["portfolio_value"],
        } for r in open_runs])
        st.plotly_chart(_equity_fig(df["Date"], df["Value"]),
                        use_container_width=True, config={"displayModeBar": False})

        _section("Daily P&L — Last 30 Sessions")
        pnl_df = pd.DataFrame([{
            "Date": pd.to_datetime(r["date"]),
            "PnL": round(r.get("daily_pnl", 0), 2),
        } for r in open_runs[-30:]])
        bar_fig = go.Figure(go.Bar(
            x=pnl_df["Date"], y=pnl_df["PnL"],
            marker_color=[C_GREEN if v >= 0 else C_RED for v in pnl_df["PnL"]],
            hovertemplate="<b>%{x}</b><br>%{y:+,.2f}<extra></extra>",
        ))
        bar_fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

    if positions:
        _section("Open Positions")
        pos_df = pd.DataFrame([{
            "Symbol":       p["symbol"],
            "Value ($)":    round(p["market_value"], 2),
            "P&L ($)":      round(p["unrealized_pl"], 2),
            "P&L (%)":      round(p["unrealized_plpc"], 2),
        } for p in positions])

        def _colour_pnl(v):
            return f"color: {C_GREEN}" if v >= 0 else f"color: {C_RED}"

        styled = (pos_df.style
                  .applymap(_colour_pnl, subset=["P&L ($)", "P&L (%)"])
                  .format({"Value ($)": "${:,.2f}", "P&L ($)": "${:+,.2f}", "P&L (%)": "{:+.2f}%"}))
        st.dataframe(styled, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TRADES
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Trades":
    st.markdown("## Trade History")

    history = load_history()
    open_runs = [r for r in history if not r["date"].endswith(("-midday", "-close"))]

    if not open_runs:
        st.info("No trade history yet.")
    else:
        all_trades   = [t for r in open_runs for t in r.get("trades_executed", [])]
        buys         = [t for t in all_trades if t.get("action") == "BUY"]
        sells        = [t for t in all_trades if "SELL" in t.get("action", "")]
        total_pnl    = sum(r.get("daily_pnl", 0) for r in open_runs)
        winning_days = sum(1 for r in open_runs if r.get("daily_pnl", 0) > 0)
        day_wr       = winning_days / len(open_runs) * 100 if open_runs else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Buys",     str(len(buys)))
        c2.metric("Total Sells",    str(len(sells)))
        c3.metric("Cumulative P&L", f"${abs(total_pnl):,.2f}", _fmt_usd(total_pnl))
        c4.metric("Winning Days",   f"{day_wr:.0f}%", f"{winning_days} of {len(open_runs)}")

        _section("All Trades")
        rows = []
        for r in reversed(open_runs):
            for t in r.get("trades_executed", []):
                rows.append({
                    "Date":   r["date"],
                    "Action": t.get("action", "?"),
                    "Symbol": t.get("symbol", "?"),
                    "Detail": t.get("detail", ""),
                    "Market": r.get("market_summary", "")[:55],
                })

        if rows:
            df = pd.DataFrame(rows)

            def _style_action(v):
                if v == "BUY":         return f"color: {C_GREEN}; font-weight: 600"
                if "SELL" in str(v):   return f"color: {C_RED}; font-weight: 600"
                return ""

            st.dataframe(
                df.style.applymap(_style_action, subset=["Action"]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No trades executed yet.")


# ══════════════════════════════════════════════════════════════════════════════
# AI DECISIONS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "AI Decisions":
    st.markdown("## AI Decision Log")
    st.caption("Every recommendation Claude made — whether executed or not")

    entries = load_decisions(n=500)
    if not entries:
        st.info("No AI decision records yet — they appear after the first trading run.")
    else:
        buy_entries  = [e for e in entries if e.get("action") == "BUY"]
        exec_entries = [e for e in entries if e.get("executed")]
        avg_conf     = sum(e.get("confidence", 0) for e in buy_entries) / len(buy_entries) if buy_entries else 0
        exec_rate    = len(exec_entries) / len(entries) * 100 if entries else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Decisions", str(len(entries)))
        c2.metric("Buy Signals",     str(len(buy_entries)))
        c3.metric("Execution Rate",  f"{exec_rate:.0f}%", f"{len(exec_entries)} executed")
        c4.metric("Avg Confidence",  f"{avg_conf:.1f} / 10")

        col_chart, col_donut = st.columns([3, 2])

        with col_chart:
            _section("Confidence Distribution (Buy Signals)")
            conf_vals = [e.get("confidence") for e in buy_entries if e.get("confidence") is not None]
            if conf_vals:
                counts = pd.Series(conf_vals).value_counts().sort_index()
                fig = go.Figure(go.Bar(
                    x=counts.index, y=counts.values,
                    marker_color=[C_ACCENT if c >= config.MIN_CONFIDENCE else C_MUTED
                                  for c in counts.index],
                    hovertemplate="Score %{x}: %{y} signals<extra></extra>",
                ))
                fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="Confidence Score")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col_donut:
            _section("Signal Types")
            sig_vals = [e.get("key_signal") for e in buy_entries if e.get("key_signal")]
            if sig_vals:
                counts = pd.Series(sig_vals).value_counts()
                pie = go.Figure(go.Pie(
                    labels=counts.index, values=counts.values,
                    hole=0.55,
                    marker_colors=[C_ACCENT, C_GREEN, C_YELLOW, C_RED, C_MUTED],
                    textfont_size=11,
                    hovertemplate="%{label}: %{value}<extra></extra>",
                ))
                pie.update_layout(**{**PLOTLY_LAYOUT, "margin": dict(l=0, r=0, t=10, b=0)})
                st.plotly_chart(pie, use_container_width=True, config={"displayModeBar": False})

        _section("Decision Log")

        f1, f2, f3 = st.columns(3)
        action_filter = f1.multiselect("Action",   ["BUY", "SELL", "HOLD"], default=["BUY", "SELL", "HOLD"])
        exec_filter   = f2.multiselect("Executed", ["Yes", "No"],           default=["Yes", "No"])
        min_conf      = f3.slider("Min Confidence", 1, 10, 1)

        filtered = [
            e for e in reversed(entries)
            if e.get("action") in action_filter
            and ("Yes" if e.get("executed") else "No") in exec_filter
            and (e.get("confidence") or 0) >= min_conf
        ]

        for e in filtered[:60]:
            action   = e.get("action", "?")
            executed = e.get("executed", False)
            symbol   = e.get("symbol", "")
            conf     = e.get("confidence", "?")
            signal   = e.get("key_signal", "")
            date     = e.get("date", "")
            reason   = e.get("reasoning", "")

            action_icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(action, "⚪")
            exec_label  = "✅ Executed" if executed else "⏭ Skipped"
            signal_txt  = f" · `{signal}`" if signal else ""

            with st.container(border=True):
                left, right = st.columns([5, 1])
                with left:
                    st.markdown(
                        f"{action_icon} **{symbol}** &nbsp; `{action}` &nbsp; {exec_label}{signal_txt}",
                        unsafe_allow_html=True,
                    )
                    if reason:
                        st.caption(reason)
                with right:
                    st.markdown(f"**{conf}**/10")
                    st.caption(date)


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Backtest":
    st.markdown("## Backtest Results")

    results = _load_backtest()
    if not results:
        st.info("No backtest results yet. Run: `python cli.py backtest --start 2024-01-01 --end 2024-12-31`")
    else:
        st.caption(
            f"Period: **{results['start']}** → **{results['end']}**  ·  "
            f"Initial capital: **${results['initial_capital']:,.2f}**"
        )

        ret = results["total_return_pct"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Return",    f"{ret:+.1f}%")
        c2.metric("Win Rate",        f"{results['win_rate_pct']:.0f}%",
                  f"{results['total_trades']} trades")
        c3.metric("Sharpe Ratio",    f"{results['sharpe_ratio']:.2f}")
        c4.metric("Max Drawdown",    f"{results['max_drawdown_pct']:.1f}%")
        c5.metric("Final Value",     f"${results['final_value']:,.2f}")

        if results.get("equity_curve"):
            _section("Equity Curve")
            eq_df = pd.DataFrame(results["equity_curve"], columns=["Date", "Value"])
            eq_df["Date"] = pd.to_datetime(eq_df["Date"])
            color = C_GREEN if ret >= 0 else C_RED
            fig = _equity_fig(eq_df["Date"], eq_df["Value"], color=color)
            fig.add_hline(
                y=results["initial_capital"],
                line_dash="dot", line_color=C_MUTED, line_width=1,
                annotation_text="Starting capital",
                annotation_font_color=C_MUTED,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        if results.get("by_signal"):
            _section("Performance by Signal")
            rows = []
            for sig, data in results["by_signal"].items():
                total = data["wins"] + data["losses"]
                wr    = data["wins"] / total * 100 if total else 0
                avg   = data["total_return"] / total if total else 0
                rows.append({
                    "Signal":       sig,
                    "Trades":       total,
                    "Win Rate":     f"{wr:.0f}%",
                    "Avg Return":   f"{avg:+.2f}%",
                    "Total Return": f"{data['total_return']:+.1f}%",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if results.get("trades"):
            with st.expander("Full trade log"):
                closed = [t for t in results["trades"]
                          if t.get("action") == "SELL" and "pnl_pct" in t]
                if closed:
                    st.dataframe(pd.DataFrame(closed), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Diagnostics":
    st.markdown("## System Diagnostics")

    report = _load_diagnostics()
    if not report:
        st.info("No diagnostic report yet. Run: `python run_diagnostics.py`")
    else:
        status  = report.get("status", "UNKNOWN")
        passed  = report.get("passed", 0)
        total   = report.get("total", 0)
        dur     = report.get("duration_seconds", 0)
        ts      = report.get("timestamp", "")

        if status == "PASS":
            st.success(f"✓  All {total} tests passing — {dur}s")
        else:
            st.error(f"✗  {passed}/{total} tests passing")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total",    str(total))
        c2.metric("Passed",   str(passed))
        c3.metric("Failed",   str(report.get("failed", 0)))
        c4.metric("Errors",   str(report.get("errors", 0)))
        c5.metric("Duration", f"{dur}s")
        st.caption(f"Last run: {ts}")

        if report.get("failures"):
            _section("Failures")
            for f in report["failures"]:
                with st.expander(f["test"]):
                    st.code(f["message"], language="text")

        st.divider()
        if st.button("Run diagnostics now", type="primary"):
            with st.spinner("Running tests..."):
                from run_diagnostics import run_diagnostics
                run_diagnostics()
                st.rerun()
