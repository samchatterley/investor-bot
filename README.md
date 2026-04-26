# InvestorBot 1.1

An autonomous AI trading bot that manages a US equities portfolio using Claude as its decision engine. It runs three times every trading day, sizes positions with Kelly Criterion, enforces a multi-layer risk framework, and conducts a weekly self-review — adjusting its own parameters based on what is and isn't working.

Built for paper trading on Alpaca Markets. Designed to move to live capital with a single `.env` change.

---

## What it does

Each trading day the bot runs three cycles:

| Time (ET) | Time (BST) | Mode | What happens |
|-----------|------------|------|--------------|
| 09:31 | 14:31 | Open | Full AI analysis → new buys + position review |
| 12:00 | 17:00 | Midday | Partial profit-taking + stop-loss sweep, no new buys |
| 15:30 | 20:30 | Close | Final position review, end-of-day summary email |

At the end of each trading day a single summary email is sent to all recipients with portfolio value, P&L, trades executed, and the AI's market commentary.

Every Sunday evening the bot runs a weekly self-review: Claude reads seven days of performance data and trade history, writes lessons learned, and applies bounded adjustments to its own configuration parameters.

---

## How it works

### AI decision layer

At the open run, the bot builds a structured prompt for Claude (`claude-sonnet-4-6`) containing:

- 30 days of price and volume data for a 28-stock universe (mega-cap tech, financials, energy, broad ETFs)
- Options chain data (put/call ratio, implied volatility)
- Macro calendar (Fed meetings, CPI, NFP dates)
- Earnings calendar (positions with earnings within 2 days are exited pre-emptively)
- Market regime classification: `BULL_TRENDING`, `CHOPPY`, `HIGH_VOL`, or `BEAR_DAY`
- Performance feedback from the signal tracking system — win rates by regime and confidence tier
- Lessons from the most recent weekly self-review

Claude returns structured decisions for each current position (hold / partial sell / full sell) and a ranked list of buy candidates with confidence scores (1–10) and reasoning.

Only candidates scoring 7 or above are acted on.

### Position sizing

Positions are sized using half-Kelly Criterion against a rolling win-rate estimate. Hard limits apply regardless:

- Maximum 5 open positions simultaneously
- Maximum 45% of portfolio in any single position
- Always retain 10% cash reserve
- Maximum $50,000 per individual order
- Maximum $150,000 total daily notional deployed

### Risk management

- **Trailing stops**: Alpaca-native trailing stop orders placed at entry (default 4% trail)
- **Partial profit taking**: Half the position sold when unrealised gain hits 8%
- **Take profit**: Full exit at 15%
- **Hold limit**: Positions auto-exit after 3 trading days regardless of P&L
- **Sector cap**: Maximum 2 positions in any single sector at once
- **Bear filter**: No new buys when SPY drops more than 1.5% in a single session
- **VIX adjustment**: Stops widen automatically when VIX exceeds 25
- **Circuit breaker**: All buying halted if intraday drawdown breaches threshold; alert sent to owner
- **Daily loss limit**: All positions closed if daily loss limit is hit
- **Earnings guard**: Positions with earnings within 2 calendar days are exited at open

### Self-improvement

The bot tracks every trade outcome against two dimensions:

- **Regime**: which of the four market states was active at entry
- **Confidence**: the AI's stated confidence score at the time of the buy

Over time this builds per-bucket win rates that feed back into the daily prompt as directive text ("In BULL_TRENDING markets, high-confidence signals have a 72% win rate — lean into these").

On Sunday evenings Claude reviews the full week, writes explicit lessons, and may adjust up to four parameters within hard-coded safety bounds:

| Parameter | Allowed range |
|-----------|---------------|
| `MIN_CONFIDENCE` | 6 – 9 |
| `TRAILING_STOP_PCT` | 2.0% – 8.0% |
| `PARTIAL_PROFIT_PCT` | 5.0% – 20.0% |
| `MAX_HOLD_DAYS` | 2 – 7 days |

All changes are applied directly to `config.py` via bounded regex replacement and reported in the Sunday email with evidence-based reasoning. No change outside the allowed range can be applied.

---

## Setup

**Requirements:** Python 3.10+, a free [Alpaca Markets](https://alpaca.markets) account, an [Anthropic API](https://console.anthropic.com) key, and a Gmail account with an App Password.

### Option A — local (Python venv)

```bash
git clone https://github.com/samchatterley/investor-bot
cd investor-bot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in .env with your keys
python run_scheduler.py
```

### Option B — Docker (one command)

```bash
cp .env.example .env
# Fill in .env with your keys
docker-compose up -d
```

This starts two containers: the trading scheduler (`investorbot`) and the web dashboard (`investorbot-dashboard`) at `http://localhost:8501`. Logs are persisted to `./logs/` via a volume mount.

### `.env` keys

| Variable | Description |
|----------|-------------|
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | Alpaca credentials |
| `ALPACA_BASE_URL` | `https://paper-api.alpaca.markets` for paper, `https://api.alpaca.markets` for live |
| `ANTHROPIC_API_KEY` | Claude API key |
| `EMAIL_FROM` | Gmail address the bot sends from |
| `EMAIL_TO` | Owner address — emergency alerts only |
| `EMAIL_RECIPIENTS` | Named recipients for daily summary + weekly review: `Sam:sam@gmail.com,Harri:harri@outlook.com` |
| `EMAIL_APP_PASSWORD` | Gmail App Password (not your login password) |

### CLI

All bot operations are available through a single CLI:

```bash
python cli.py status              # Account value, open positions, halt state
python cli.py positions           # Live positions with P&L
python cli.py trades --days 10    # Recent trade history
python cli.py decisions --days 5  # AI decision log with reasoning
python cli.py run --mode open     # Trigger a trading run
python cli.py run --dry-run       # Analyse only, no orders placed
python cli.py halt                # Emergency kill switch
python cli.py resume              # Clear halt and resume
python cli.py backtest --start 2025-01-01 --end 2025-12-31
python cli.py dashboard           # Launch web dashboard
```

### Web dashboard

```bash
python cli.py dashboard
# or directly:
streamlit run dashboard.py
```

Opens at `http://localhost:8501`. Five pages:

| Page | Contents |
|------|----------|
| Overview | Live portfolio value, equity curve, daily P&L bar chart, open positions |
| Trades | Full trade history table across all sessions |
| AI Decisions | Every Claude recommendation — confidence, signal type, reasoning, executed flag |
| Backtest | Equity curve, Sharpe ratio, win rate, signal breakdown |
| Diagnostics | Unit test results with pass/fail counts and a run-now button |

### Backtesting

```bash
python cli.py backtest --start 2025-01-01 --end 2025-12-31 --capital 25000
```

Replays the strategy's rule-based entry signals on historical OHLCV data without calling Claude (no API cost). Reports total return, win rate, Sharpe ratio, max drawdown, and performance by signal type. Results are saved to `logs/backtest_results.json` and rendered in the dashboard.

### Kill switch

```bash
python cli.py halt          # Interactive kill switch (prompts for confirmation)
python cli.py resume        # Clear halt and resume
```

---

## Notifications

Each person listed in `EMAIL_RECIPIENTS` receives a personalised email addressed by name.

| Event | Recipients |
|-------|-----------|
| End-of-day summary | All `EMAIL_RECIPIENTS` |
| Sunday weekly review + diagnostics | All `EMAIL_RECIPIENTS` |
| Circuit breaker / daily loss limit / errors | `EMAIL_TO` only |

---

## Notes of interest

- **Paper-first by default.** The `ALPACA_BASE_URL` in `.env.example` points to the paper trading endpoint. Switching to live requires only changing that one value.

- **Fractional shares.** All orders use fractional share support, so the full calculated dollar amount is deployed rather than rounding down to whole shares. This matters most for high-price names like NVDA or GOOGL.

- **Dependencies are version-pinned.** `requirements.txt` pins exact versions to prevent silent behaviour changes from upstream updates. Test in paper mode before upgrading any dependency.

- **Logs stay local.** The `logs/` directory is gitignored and never leaves the machine. Each run writes a timestamped JSON record; `get_day_summary()` merges the three daily records into one end-of-day view for the email.

- **MiFID II-style pre-trade controls.** The fat-finger guard (`MAX_SINGLE_ORDER_USD`) and runaway algorithm guard (`MAX_DAILY_NOTIONAL_USD`) are modelled on Article 17 algorithmic trading obligations — belt-and-braces limits that apply regardless of what Claude decides.

- **AI explainability.** Every recommendation Claude makes is logged to `logs/decisions.jsonl` with its confidence score, plain-English reasoning, and signal type — whether or not the trade was ultimately executed. This log feeds the AI Decisions page in the dashboard.

- **203 unit tests.** The test suite covers all core logic modules and runs automatically every Sunday as part of the weekly review job. Results are included in the email and visible in the Diagnostics dashboard page.

---

## Version history

### 1.1 — April 2026
Added web dashboard (Streamlit, 5 pages), CLI (`cli.py`), Docker deploy (`docker-compose up`), AI decision log (`logs/decisions.jsonl`), personalised email greetings per recipient, Sharpe ratio in backtester, and backtest results persisted for the dashboard. Unit test suite expanded to 203 tests.

### 1.0 — April 2026
Initial release. Full autonomous paper-trading capability with AI-driven decision making, Kelly Criterion sizing, multi-layer risk management, regime-aware signal tracking, weekly self-review with bounded self-modification, and multi-recipient email reporting.
