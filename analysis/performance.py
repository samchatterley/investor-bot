import json
import logging
import math
import os
from datetime import date

from config import LOG_DIR

logger = logging.getLogger(__name__)

_STATS_PATH = os.path.join(LOG_DIR, "signal_stats.json")
_DASHBOARD_PATH = os.path.join(LOG_DIR, "dashboard.html")


# ---------- Signal win-rate tracking ----------

def _load_stats() -> dict:
    os.makedirs(LOG_DIR, exist_ok=True)
    if os.path.exists(_STATS_PATH):
        with open(_STATS_PATH) as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def _save_stats(stats: dict):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)


def _empty_bucket() -> dict:
    return {"trades": 0, "wins": 0, "losses": 0, "total_return_pct": 0.0}


def _update_bucket(bucket: dict, return_pct: float):
    bucket["trades"] += 1
    bucket["total_return_pct"] = round(bucket["total_return_pct"] + return_pct, 4)
    if return_pct > 0:
        bucket["wins"] += 1
    else:
        bucket["losses"] += 1


def record_trade_outcome(signal: str, return_pct: float,
                         regime: str = "UNKNOWN", confidence: int = 0):
    """
    Record the outcome of a closed trade against its entry signal, regime, and confidence.
    Called when a position is closed (sell, stop loss, stale exit).
    """
    stats = _load_stats()
    if signal not in stats:
        stats[signal] = {**_empty_bucket(), "by_regime": {}, "by_confidence": {}}

    entry = stats[signal]
    entry.setdefault("by_regime", {})
    entry.setdefault("by_confidence", {})

    _update_bucket(entry, return_pct)

    entry["by_regime"].setdefault(regime, _empty_bucket())
    _update_bucket(entry["by_regime"][regime], return_pct)

    conf_key = str(confidence) if confidence else "unknown"
    entry["by_confidence"].setdefault(conf_key, _empty_bucket())
    _update_bucket(entry["by_confidence"][conf_key], return_pct)

    _save_stats(stats)
    logger.info(f"Signal stats updated: {signal} [regime={regime} conf={confidence}] → {return_pct:+.2f}%")


def _bucket_summary(bucket: dict) -> dict:
    t = bucket.get("trades", 0)
    if t == 0:
        return {"trades": 0, "win_rate": 0.0, "avg_return_pct": 0.0}
    return {
        "trades": t,
        "win_rate": round(bucket["wins"] / t * 100, 1),
        "avg_return_pct": round(bucket["total_return_pct"] / t, 2),
    }


def get_win_rates() -> dict[str, dict]:
    """Return win rates and average return by signal type, with regime and confidence breakdowns."""
    stats = _load_stats()
    result = {}
    for signal, data in stats.items():
        if data.get("trades", 0) == 0:
            continue
        result[signal] = {
            **_bucket_summary(data),
            "by_regime": {
                regime: _bucket_summary(b)
                for regime, b in data.get("by_regime", {}).items()
                if b.get("trades", 0) > 0
            },
            "by_confidence": {
                conf: _bucket_summary(b)
                for conf, b in data.get("by_confidence", {}).items()
                if b.get("trades", 0) > 0
            },
        }
    return result


def get_actionable_feedback() -> str:
    """
    Generate directive text from accumulated signal stats for inclusion in the AI prompt.
    Only surfaces meaningful conclusions — requires at least 3 trades per signal.
    """
    win_rates = get_win_rates()
    if not win_rates:
        return ""

    lines = []
    for signal, data in win_rates.items():
        if data["trades"] < 3:
            continue
        wr = data["win_rate"]
        avg = data["avg_return_pct"]

        if wr >= 60:
            verdict = f"working well ({wr:.0f}% win rate, avg {avg:+.2f}%) — maintain confidence scoring"
        elif wr >= 45:
            verdict = f"marginal ({wr:.0f}% win rate, avg {avg:+.2f}%) — be selective, prefer higher setups"
        else:
            verdict = f"underperforming ({wr:.0f}% win rate, avg {avg:+.2f}%) — raise required confidence by 1"

        line = f"  {signal}: {verdict}"

        regime_notes = []
        for regime, rdata in data.get("by_regime", {}).items():
            if rdata["trades"] >= 2:
                if rdata["win_rate"] >= 65:
                    regime_notes.append(f"{regime} strong ({rdata['win_rate']:.0f}%)")
                elif rdata["win_rate"] < 40:
                    regime_notes.append(f"{regime} avoid ({rdata['win_rate']:.0f}%)")
        if regime_notes:
            line += f" | {', '.join(regime_notes)}"

        lines.append(line)

    if not lines:
        return ""

    return (
        "PERFORMANCE FEEDBACK — adjust confidence scoring based on what has actually worked:\n"
        + "\n".join(lines) + "\n"
    )


# ---------- HTML dashboard ----------

def compute_metrics(records: list[dict]) -> dict:
    if not records:
        return {}

    values = [r["account_after"]["portfolio_value"] for r in records]
    pnls = [r.get("daily_pnl", 0) for r in records]

    peak = values[0]
    max_drawdown = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (v - peak) / peak * 100 if peak > 0 else 0
        if dd < max_drawdown:
            max_drawdown = dd

    total_return = (values[-1] / values[0] - 1) * 100 if values[0] > 0 else 0
    winning_days = sum(1 for p in pnls if p > 0)
    win_rate = winning_days / len(pnls) * 100 if pnls else 0

    # Simplified Sharpe: mean daily return / std dev, annualised
    # Use account_before as denominator — return = pnl / starting_value for that day
    daily_returns = [
        r.get("daily_pnl", 0) / r["account_before"]["portfolio_value"]
        for r in records
        if r["account_before"]["portfolio_value"] > 0
    ]
    if len(daily_returns) > 1:
        mean_r = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_r) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        std_r = math.sqrt(variance) if variance > 0 else 1e-9
        sharpe = (mean_r / std_r) * math.sqrt(252)
    else:
        sharpe = 0.0

    return {
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "win_rate_pct": round(win_rate, 1),
        "sharpe": round(sharpe, 2),
        "total_trades": sum(len(r.get("trades_executed", [])) for r in records),
        "days_traded": len(records),
    }


def generate_dashboard(records: list[dict]):
    """Generate an HTML performance dashboard at logs/dashboard.html."""
    if not records:
        return

    metrics = compute_metrics(records)
    win_rates = get_win_rates()

    dates_js = ", ".join(f'"{r["date"]}"' for r in records)
    values_js = ", ".join(str(r["account_after"]["portfolio_value"]) for r in records)
    pnl_js = ", ".join(str(round(r.get("daily_pnl", 0), 2)) for r in records)

    signal_labels = list(win_rates.keys())
    signal_win_rates = [win_rates[s]["win_rate"] for s in signal_labels]
    signal_avg_returns = [win_rates[s]["avg_return_pct"] for s in signal_labels]

    signal_labels_js = ", ".join(f'"{s}"' for s in signal_labels)
    signal_wr_js = ", ".join(str(w) for w in signal_win_rates)
    signal_ar_js = ", ".join(str(r) for r in signal_avg_returns)

    trade_rows = ""
    for r in reversed(records[-30:]):
        for t in r.get("trades_executed", []):
            action = t.get("action", "?")
            colour = "#2e7d32" if action == "BUY" else "#c62828"
            trade_rows += f"""
            <tr>
                <td>{r['date']}</td>
                <td><b style="color:{colour}">{action}</b></td>
                <td>{t.get('symbol','')}</td>
                <td>{t.get('detail','')}</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Trading Bot Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body {{ font-family: -apple-system, Arial, sans-serif; background: #f5f5f5; margin: 0; padding: 24px; color: #222; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .subtitle {{ color: #888; font-size: 13px; margin-bottom: 24px; }}
  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }}
  .card {{ background: #fff; border-radius: 10px; padding: 16px 20px; flex: 1; min-width: 120px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  .card .label {{ font-size: 11px; color: #aaa; text-transform: uppercase; letter-spacing: .5px; }}
  .card .value {{ font-size: 26px; font-weight: 700; margin-top: 4px; }}
  .chart-wrap {{ background: #fff; border-radius: 10px; padding: 20px; margin-bottom: 24px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  .chart-wrap h2 {{ font-size: 15px; margin: 0 0 16px; }}
  .charts-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ padding: 8px; background: #f0f0f0; text-align: left; }}
  td {{ padding: 8px; border-bottom: 1px solid #eee; }}
  .pos {{ color: #2e7d32; }} .neg {{ color: #c62828; }}
</style>
</head>
<body>
<h1>Trading Bot Performance</h1>
<p class="subtitle">Generated {date.today().isoformat()} · {metrics.get('days_traded',0)} days of data</p>

<div class="cards">
  <div class="card">
    <div class="label">Total Return</div>
    <div class="value {'pos' if metrics.get('total_return_pct',0) >= 0 else 'neg'}">{metrics.get('total_return_pct',0):+.1f}%</div>
  </div>
  <div class="card">
    <div class="label">Win Rate</div>
    <div class="value">{metrics.get('win_rate_pct',0):.0f}%</div>
  </div>
  <div class="card">
    <div class="label">Max Drawdown</div>
    <div class="value neg">{metrics.get('max_drawdown_pct',0):.1f}%</div>
  </div>
  <div class="card">
    <div class="label">Sharpe Ratio</div>
    <div class="value">{metrics.get('sharpe',0):.2f}</div>
  </div>
  <div class="card">
    <div class="label">Total Trades</div>
    <div class="value">{metrics.get('total_trades',0)}</div>
  </div>
</div>

<div class="chart-wrap">
  <h2>Portfolio Value</h2>
  <canvas id="equityChart" height="80"></canvas>
</div>

<div class="charts-row">
  <div class="chart-wrap">
    <h2>Daily P&L</h2>
    <canvas id="pnlChart"></canvas>
  </div>
  <div class="chart-wrap">
    <h2>Win Rate by Signal</h2>
    <canvas id="signalChart"></canvas>
  </div>
</div>

<div class="chart-wrap">
  <h2>Recent Trades</h2>
  <table>
    <thead><tr><th>Date</th><th>Action</th><th>Symbol</th><th>Detail</th></tr></thead>
    <tbody>{trade_rows}</tbody>
  </table>
</div>

<script>
const dates = [{dates_js}];
const values = [{values_js}];
const pnls = [{pnl_js}];
const signalLabels = [{signal_labels_js}];
const signalWR = [{signal_wr_js}];
const signalAR = [{signal_ar_js}];

new Chart(document.getElementById('equityChart'), {{
  type: 'line',
  data: {{ labels: dates, datasets: [{{ label: 'Portfolio Value ($)', data: values, borderColor: '#1565c0', fill: true, backgroundColor: 'rgba(21,101,192,.08)', tension: 0.3, pointRadius: 2 }}] }},
  options: {{ plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ beginAtZero: false }} }} }}
}});

new Chart(document.getElementById('pnlChart'), {{
  type: 'bar',
  data: {{ labels: dates, datasets: [{{ label: 'Daily P&L ($)', data: pnls, backgroundColor: pnls.map(v => v >= 0 ? '#2e7d32' : '#c62828') }}] }},
  options: {{ plugins: {{ legend: {{ display: false }} }} }}
}});

{'new Chart(document.getElementById("signalChart"), { type: "bar", data: { labels: signalLabels, datasets: [{ label: "Win Rate %", data: signalWR, backgroundColor: "#1565c0" }] }, options: { scales: { y: { max: 100 } } } });' if signal_labels else ''}
</script>
</body>
</html>"""

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(_DASHBOARD_PATH, "w") as f:
        f.write(html)
    logger.info(f"Dashboard saved to {_DASHBOARD_PATH}")
