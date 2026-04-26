"""
InvestorBot Web Dashboard

Launch:  python cli.py dashboard
         streamlit run dashboard.py
"""

import json
import os

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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

C_BG        = "#0b0f1a"
C_SURFACE   = "#131929"
C_SURFACE2  = "#1a2236"
C_BORDER    = "rgba(255,255,255,0.08)"
C_ACCENT    = "#00d4aa"
C_GREEN     = "#26c281"
C_RED       = "#e05c5c"
C_YELLOW    = "#f0b429"
C_TEXT      = "#e2e8f0"
C_MUTED     = "#8896a5"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=C_TEXT, family="system-ui, -apple-system, sans-serif"),
    margin=dict(l=0, r=0, t=28, b=0),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        showline=False,
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.05)",
        showline=False,
        zeroline=False,
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,0,0,0)",
    ),
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}}

/* Hide default Streamlit chrome */
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: {C_SURFACE};
    border-right: 1px solid {C_BORDER};
}}
[data-testid="stSidebar"] .stRadio label {{
    font-size: 0.9rem;
    padding: 6px 0;
}}

/* Metric cards */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 14px;
    margin-bottom: 1.5rem;
}}
.kpi-card {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 12px;
    padding: 20px 18px 16px;
    position: relative;
    overflow: hidden;
}}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: {C_ACCENT};
    border-radius: 12px 12px 0 0;
}}
.kpi-card.green::before {{ background: {C_GREEN}; }}
.kpi-card.red::before {{ background: {C_RED}; }}
.kpi-card.yellow::before {{ background: {C_YELLOW}; }}
.kpi-label {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {C_MUTED};
    margin-bottom: 8px;
}}
.kpi-value {{
    font-size: 1.55rem;
    font-weight: 700;
    color: {C_TEXT};
    line-height: 1.1;
}}
.kpi-sub {{
    font-size: 0.78rem;
    color: {C_MUTED};
    margin-top: 6px;
}}
.kpi-delta {{
    font-size: 0.82rem;
    font-weight: 600;
    margin-top: 6px;
}}
.kpi-delta.pos {{ color: {C_GREEN}; }}
.kpi-delta.neg {{ color: {C_RED}; }}

/* Section headers */
.section-header {{
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {C_MUTED};
    margin: 1.8rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.section-header::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: {C_BORDER};
}}

/* Status pill */
.pill {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}}
.pill-green {{ background: rgba(38,194,129,0.15); color: {C_GREEN}; border: 1px solid rgba(38,194,129,0.3); }}
.pill-red   {{ background: rgba(224,92,92,0.15);  color: {C_RED};   border: 1px solid rgba(224,92,92,0.3); }}
.pill-teal  {{ background: rgba(0,212,170,0.12);  color: {C_ACCENT}; border: 1px solid rgba(0,212,170,0.25); }}
.pill-yellow {{ background: rgba(240,180,41,0.12); color: {C_YELLOW}; border: 1px solid rgba(240,180,41,0.25); }}

/* Positions table */
.pos-table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
.pos-table th {{
    text-align: left;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {C_MUTED};
    padding: 8px 12px;
    border-bottom: 1px solid {C_BORDER};
}}
.pos-table td {{
    padding: 10px 12px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    color: {C_TEXT};
}}
.pos-table tr:last-child td {{ border-bottom: none; }}
.pos-table tr:hover td {{ background: rgba(255,255,255,0.02); }}
.num {{ font-variant-numeric: tabular-nums; }}
.pos  {{ color: {C_GREEN}; }}
.neg  {{ color: {C_RED};   }}

/* Decision log cards */
.decision-card {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}}
.decision-card.executed {{ border-left: 3px solid {C_ACCENT}; }}
.decision-card.not-executed {{ border-left: 3px solid {C_BORDER}; }}
.decision-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}}
.decision-symbol {{ font-size: 1rem; font-weight: 700; color: {C_TEXT}; }}
.decision-reasoning {{ font-size: 0.83rem; color: {C_MUTED}; line-height: 1.5; margin-top: 4px; }}

/* Chart containers */
.chart-container {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 1rem;
}}

/* Diagnostics */
.diag-pass {{
    background: rgba(38,194,129,0.1);
    border: 1px solid rgba(38,194,129,0.3);
    border-radius: 10px;
    padding: 16px 20px;
    font-size: 0.95rem;
    color: {C_GREEN};
}}
.diag-fail {{
    background: rgba(224,92,92,0.1);
    border: 1px solid rgba(224,92,92,0.3);
    border-radius: 10px;
    padding: 16px 20px;
    font-size: 0.95rem;
    color: {C_RED};
}}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _kpi(label: str, value: str, sub: str = "", delta: str = "", accent: str = "default") -> str:
    delta_html = ""
    if delta:
        cls = "pos" if delta.startswith("+") else "neg"
        delta_html = f'<div class="kpi-delta {cls}">{delta}</div>'
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    card_cls = {"green": "green", "red": "red", "yellow": "yellow"}.get(accent, "")
    return f"""
    <div class="kpi-card {card_cls}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}{sub_html}
    </div>"""


def _pill(text: str, style: str = "teal") -> str:
    return f'<span class="pill pill-{style}">{text}</span>'


def _section(title: str) -> None:
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def _pnl_color(v: float) -> str:
    return C_GREEN if v >= 0 else C_RED


def _fmt_pct(v: float) -> str:
    return f"+{v:.2f}%" if v >= 0 else f"{v:.2f}%"


def _fmt_usd(v: float) -> str:
    return f"+${v:,.2f}" if v >= 0 else f"-${abs(v):,.2f}"


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


def _equity_fig(dates, values, color=C_ACCENT, title=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=values,
        mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
        hovertemplate="<b>%{x}</b><br>$%{y:,.2f}<extra></extra>",
    ))
    layout = {**PLOTLY_LAYOUT, "title": dict(text=title, font=dict(size=13, color=C_MUTED)) if title else {}}
    fig.update_layout(**layout)
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

halted = os.path.exists(config.HALT_FILE)
mode_label = "PAPER" if config.IS_PAPER else "LIVE"

with st.sidebar:
    st.markdown("### 📈 InvestorBot")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Overview", "Trades", "AI Decisions", "Backtest", "Diagnostics"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    status_pill = _pill("HALTED", "red") if halted else _pill("Active", "green")
    mode_pill   = _pill("LIVE", "red") if not config.IS_PAPER else _pill("PAPER", "teal")
    st.markdown(
        f"**Status** &nbsp; {status_pill}<br><br>**Mode** &nbsp;&nbsp;&nbsp; {mode_pill}",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "Overview":
    st.markdown("## Portfolio Overview")

    acc, positions, err = _load_account()
    history = load_history()
    open_runs = [r for r in history if not r["date"].endswith(("-midday", "-close"))]

    # KPI row
    if acc:
        pnl_today = open_runs[-1].get("daily_pnl", 0) if open_runs else 0
        invested   = acc["portfolio_value"] - acc["cash"]
        invested_pct = (invested / acc["portfolio_value"] * 100) if acc["portfolio_value"] else 0

        cards = (
            _kpi("Portfolio Value", f"${acc['portfolio_value']:,.2f}",
                 delta=_fmt_usd(pnl_today),
                 accent="green" if pnl_today >= 0 else "red") +
            _kpi("Cash", f"${acc['cash']:,.2f}",
                 sub=f"{100 - invested_pct:.0f}% of portfolio") +
            _kpi("Invested", f"${invested:,.2f}",
                 sub=f"{invested_pct:.0f}% deployed") +
            _kpi("Positions", f"{len(positions)} / {config.MAX_POSITIONS}",
                 sub="open right now",
                 accent="yellow" if len(positions) >= config.MAX_POSITIONS else "default")
        )
        st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)
    elif err:
        st.warning(f"Could not connect to Alpaca ({err}) — showing historical data only.")

    # Equity curve
    if open_runs:
        _section("Equity Curve")
        df = pd.DataFrame([
            {"Date": pd.to_datetime(r["date"]),
             "Value": r["account_after"]["portfolio_value"]}
            for r in open_runs
        ])
        fig = _equity_fig(df["Date"], df["Value"])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # P&L bars
        _section("Daily P&L — Last 30 Sessions")
        pnl_data = [
            {"Date": pd.to_datetime(r["date"]), "PnL": round(r.get("daily_pnl", 0), 2)}
            for r in open_runs[-30:]
        ]
        pnl_df = pd.DataFrame(pnl_data)
        bar_fig = go.Figure(go.Bar(
            x=pnl_df["Date"], y=pnl_df["PnL"],
            marker_color=[C_GREEN if v >= 0 else C_RED for v in pnl_df["PnL"]],
            hovertemplate="<b>%{x}</b><br>%{y:+,.2f}<extra></extra>",
        ))
        bar_fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

    # Open positions
    if positions:
        _section("Open Positions")
        rows = ""
        for p in positions:
            pl    = p["unrealized_pl"]
            pct   = p["unrealized_plpc"]
            cls   = "pos" if pl >= 0 else "neg"
            rows += f"""
            <tr>
                <td><strong>{p['symbol']}</strong></td>
                <td class="num">${p['market_value']:,.2f}</td>
                <td class="num {cls}">{_fmt_usd(pl)}</td>
                <td class="num {cls}">{_fmt_pct(pct)}</td>
            </tr>"""
        st.markdown(f"""
        <table class="pos-table">
            <thead><tr>
                <th>Symbol</th><th>Value</th><th>Unrealised P&L</th><th>P&L %</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
        """, unsafe_allow_html=True)


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
        # Summary KPIs
        all_trades = [t for r in open_runs for t in r.get("trades_executed", [])]
        buys  = [t for t in all_trades if t.get("action") == "BUY"]
        sells = [t for t in all_trades if t.get("action") in ("SELL", "SELL_PARTIAL")]
        total_pnl = sum(r.get("daily_pnl", 0) for r in open_runs)
        winning_days = sum(1 for r in open_runs if r.get("daily_pnl", 0) > 0)
        day_wr = winning_days / len(open_runs) * 100 if open_runs else 0

        cards = (
            _kpi("Total Buys",  str(len(buys)),  sub="orders placed") +
            _kpi("Total Sells", str(len(sells)), sub="exits") +
            _kpi("Cumulative P&L", f"${abs(total_pnl):,.2f}",
                 delta=_fmt_usd(total_pnl),
                 accent="green" if total_pnl >= 0 else "red") +
            _kpi("Winning Days", f"{day_wr:.0f}%",
                 sub=f"{winning_days} of {len(open_runs)} days")
        )
        st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)

        # Trade table
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

            # Colour the Action column
            def _style_action(val):
                if val == "BUY":
                    return f"color: {C_GREEN}; font-weight: 600"
                elif "SELL" in val:
                    return f"color: {C_RED}; font-weight: 600"
                return ""

            styled = df.style.applymap(_style_action, subset=["Action"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
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
        # KPIs
        buy_entries  = [e for e in entries if e.get("action") == "BUY"]
        exec_entries = [e for e in entries if e.get("executed")]
        avg_conf     = sum(e.get("confidence", 0) for e in buy_entries) / len(buy_entries) if buy_entries else 0
        exec_rate    = len(exec_entries) / len(entries) * 100 if entries else 0

        cards = (
            _kpi("Total Decisions", str(len(entries))) +
            _kpi("Buy Signals",     str(len(buy_entries))) +
            _kpi("Executed",        f"{exec_rate:.0f}%", sub=f"{len(exec_entries)} of {len(entries)}") +
            _kpi("Avg Confidence",  f"{avg_conf:.1f} / 10")
        )
        st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            # Confidence distribution
            _section("Confidence Distribution (Buy Signals)")
            conf_vals = [e.get("confidence") for e in buy_entries if e.get("confidence") is not None]
            if conf_vals:
                counts = pd.Series(conf_vals).value_counts().sort_index()
                conf_fig = go.Figure(go.Bar(
                    x=counts.index, y=counts.values,
                    marker_color=[C_ACCENT if c >= config.MIN_CONFIDENCE else C_MUTED for c in counts.index],
                    hovertemplate="Confidence %{x}: %{y} signals<extra></extra>",
                ))
                conf_fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="Confidence Score", yaxis_title="Count")
                st.plotly_chart(conf_fig, use_container_width=True, config={"displayModeBar": False})

        with col2:
            # Signal type breakdown
            _section("Signal Types")
            sig_vals = [e.get("key_signal") for e in buy_entries if e.get("key_signal")]
            if sig_vals:
                sig_counts = pd.Series(sig_vals).value_counts()
                pie_fig = go.Figure(go.Pie(
                    labels=sig_counts.index,
                    values=sig_counts.values,
                    hole=0.55,
                    marker_colors=[C_ACCENT, C_GREEN, C_YELLOW, C_RED, C_MUTED],
                    textfont_size=11,
                    hovertemplate="%{label}: %{value} signals<extra></extra>",
                ))
                pie_fig.update_layout(**{**PLOTLY_LAYOUT, "margin": dict(l=0, r=0, t=10, b=0)})
                st.plotly_chart(pie_fig, use_container_width=True, config={"displayModeBar": False})

        # Filters
        _section("Decision Log")
        f1, f2, f3 = st.columns(3)
        action_filter = f1.multiselect("Action", ["BUY", "SELL", "HOLD"], default=["BUY", "SELL", "HOLD"])
        exec_filter   = f2.multiselect("Executed", ["Yes", "No"], default=["Yes", "No"])
        min_conf      = f3.slider("Min Confidence", 1, 10, 1)

        filtered = [
            e for e in reversed(entries)
            if e.get("action") in action_filter
            and ("Yes" if e.get("executed") else "No") in exec_filter
            and (e.get("confidence") or 0) >= min_conf
        ]

        for e in filtered[:80]:
            executed    = e.get("executed", False)
            action      = e.get("action", "?")
            conf        = e.get("confidence", "?")
            signal      = e.get("key_signal", "")
            reasoning   = e.get("reasoning", "")
            symbol      = e.get("symbol", "")
            date        = e.get("date", "")

            if action == "BUY":
                action_pill = _pill("BUY", "green")
            elif "SELL" in action:
                action_pill = _pill(action, "red")
            else:
                action_pill = _pill("HOLD", "yellow")

            exec_pill  = _pill("Executed", "teal") if executed else _pill("Skipped", "yellow")
            signal_html = _pill(signal, "teal") if signal else ""

            st.markdown(f"""
            <div class="decision-card {'executed' if executed else 'not-executed'}">
                <div class="decision-header">
                    <span class="decision-symbol">{symbol}</span>
                    {action_pill} {exec_pill} {signal_html}
                    <span style="margin-left:auto;font-size:0.75rem;color:{C_MUTED};">{date} &nbsp;·&nbsp; conf {conf}/10</span>
                </div>
                <div class="decision-reasoning">{reasoning}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Backtest":
    st.markdown("## Backtest Results")

    results = _load_backtest()
    if not results:
        st.info("No backtest results yet.  Run: `python cli.py backtest --start 2024-01-01 --end 2024-12-31`")
    else:
        st.caption(f"Period: **{results['start']}** → **{results['end']}**  ·  Initial capital: **${results['initial_capital']:,.2f}**")

        ret = results["total_return_pct"]
        cards = (
            _kpi("Total Return",     f"{ret:+.1f}%",       accent="green" if ret >= 0 else "red") +
            _kpi("Win Rate",         f"{results['win_rate_pct']:.0f}%",
                 sub=f"{results['total_trades']} trades") +
            _kpi("Sharpe Ratio",     f"{results['sharpe_ratio']:.2f}",
                 accent="green" if results['sharpe_ratio'] >= 1 else "yellow") +
            _kpi("Max Drawdown",     f"{results['max_drawdown_pct']:.1f}%", accent="red") +
            _kpi("Final Value",      f"${results['final_value']:,.2f}")
        )
        st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)

        # Equity curve
        if results.get("equity_curve"):
            _section("Equity Curve")
            eq_df = pd.DataFrame(results["equity_curve"], columns=["Date", "Value"])
            eq_df["Date"] = pd.to_datetime(eq_df["Date"])
            color = C_GREEN if results["total_return_pct"] >= 0 else C_RED
            fig = _equity_fig(eq_df["Date"], eq_df["Value"], color=color)
            fig.add_hline(
                y=results["initial_capital"],
                line_dash="dot", line_color=C_MUTED, line_width=1,
                annotation_text="Starting capital",
                annotation_font_color=C_MUTED,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Signal breakdown
        if results.get("by_signal"):
            _section("Performance by Signal")
            sig_rows = []
            for sig, data in results["by_signal"].items():
                total = data["wins"] + data["losses"]
                wr    = data["wins"] / total * 100 if total else 0
                avg   = data["total_return"] / total if total else 0
                sig_rows.append({
                    "Signal": sig,
                    "Trades": total,
                    "Win Rate": f"{wr:.0f}%",
                    "Avg Return": f"{avg:+.2f}%",
                    "Total Return": f"{data['total_return']:+.1f}%",
                })
            st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)

        # Full trade log
        if results.get("trades"):
            with st.expander("Full trade log"):
                closed = [t for t in results["trades"] if t.get("action") == "SELL" and "pnl_pct" in t]
                if closed:
                    trade_df = pd.DataFrame(closed)
                    st.dataframe(trade_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Diagnostics":
    st.markdown("## System Diagnostics")

    report = _load_diagnostics()
    if not report:
        st.info("No diagnostic report yet. Run: `python run_diagnostics.py`")
    else:
        ts     = report.get("timestamp", "")
        status = report.get("status", "UNKNOWN")
        passed = report.get("passed", 0)
        total  = report.get("total", 0)
        dur    = report.get("duration_seconds", 0)

        if status == "PASS":
            st.markdown(
                f'<div class="diag-pass">✓ &nbsp; All {total} tests passing &nbsp;·&nbsp; {dur}s</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="diag-fail">✗ &nbsp; {passed}/{total} tests passing</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        cards = (
            _kpi("Total",   str(total)) +
            _kpi("Passed",  str(passed),  accent="green") +
            _kpi("Failed",  str(report.get("failed", 0)),  accent="red" if report.get("failed") else "default") +
            _kpi("Errors",  str(report.get("errors", 0)),  accent="red" if report.get("errors") else "default") +
            _kpi("Duration", f"{dur}s")
        )
        st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)

        st.caption(f"Last run: {ts}")

        if report.get("failures"):
            _section("Failures")
            for f in report["failures"]:
                with st.expander(f["test"]):
                    st.code(f["message"], language="text")

        st.markdown("---")
        if st.button("Run diagnostics now", type="primary"):
            with st.spinner("Running 203 tests..."):
                from run_diagnostics import run_diagnostics
                run_diagnostics()
                st.rerun()
