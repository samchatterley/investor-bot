# InvestorBot Live Runbook

Operations guide for going live with real capital. Read in full before first live session.

---

## Contents

1. [Pre-Live Checklist](#1-pre-live-checklist)
2. [Canary Procedure](#2-canary-procedure)
3. [Going Live](#3-going-live)
4. [Monitoring During Runs](#4-monitoring-during-runs)
5. [Incident Response](#5-incident-response)
6. [Emergency Halt](#6-emergency-halt)

---

## 1. Pre-Live Checklist

Complete every item before advancing to the canary procedure. Do not skip steps.

### 1.1 Account and API

- [ ] Alpaca live account funded and approved for trading
- [ ] Live API keys set in `.env` (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_BASE_URL=https://api.alpaca.markets`)
- [ ] `TRADING_MODE=live` set in `.env`
- [ ] `LIVE_CONFIRM=I-ACCEPT-REAL-MONEY-RISK` set in `.env`
- [ ] `ANTHROPIC_API_KEY` set and credited

### 1.2 Safety check — must be GREEN

```bash
python main.py --safety-check
```

Expected output: `Result: BROKER_HEALTH=GREEN`. Do not continue if YELLOW or RED.

### 1.3 Paper run verified (minimum 3 sessions)

- [ ] At least 3 paper sessions completed with no unhandled exceptions in `logs/scheduler.log`
- [ ] No HALT file created unexpectedly
- [ ] AI decisions pass validation on every run (`validate_ai_response` reported no structural errors)
- [ ] Stop coverage confirmed after every buy (`ORDER_TIMING` events in audit log)

### 1.4 Shadow run passed

```bash
python main.py --live-shadow --mode open
```

Verify in `logs/scheduler.log`:
- `LIVE_SHADOW_START` event logged
- `WOULD_BUY` events appear with sizing and signal data
- `LIVE_SHADOW_COMPLETE` event logged with non-zero `would_buy_count`
- No exceptions, no HALT file

### 1.5 Config review

- [ ] `SMALL_ACCOUNT_MODE=true` if account is < $500
- [ ] `MAX_SINGLE_ORDER_USD` set to a value you are comfortable losing entirely on day one
- [ ] `MAX_DAILY_NOTIONAL_USD` ≤ `MAX_SINGLE_ORDER_USD` for canary session
- [ ] `MAX_POSITIONS=1` for canary session
- [ ] `MAX_EXPERIMENT_DRAWDOWN_USD` set to the most you are willing to lose over the full experiment

---

## 2. Canary Procedure

A canary run is a single trade with the smallest viable notional to prove the live broker pipeline end-to-end before committing meaningful capital. Success means a real order was placed, filled, stop-protected, and manually closed — with every event appearing in the audit log.

### 2.1 Canary config

Copy `canary.env.example` to `.env` and fill in your live API credentials:

```bash
cp canary.env.example .env
# edit .env — add ALPACA_API_KEY, ALPACA_SECRET_KEY, ANTHROPIC_API_KEY
```

Key parameters in `canary.env.example`:

| Variable | Canary value | Purpose |
|----------|-------------|---------|
| `TRADING_MODE` | `live` | Enables live broker |
| `SMALL_ACCOUNT_MODE` | `true` | Conservative sizing defaults |
| `MAX_POSITIONS` | `1` | One position only |
| `MAX_SINGLE_ORDER_USD` | `20` | Maximum order size |
| `MAX_DAILY_NOTIONAL_USD` | `20` | Cap total deployment |
| `MAX_DEPLOYED_USD` | `20` | Cap open exposure |
| `MAX_EXPERIMENT_DRAWDOWN_USD` | `20` | Hard stop for the whole experiment |
| `MIN_CONFIDENCE` | `8` | Higher threshold — only high-conviction buys |

### 2.2 Step-by-step canary

**Step 1 — Final safety check with live credentials**

```bash
python main.py --safety-check
```

Must be GREEN. If YELLOW/RED: resolve issues before continuing.

**Step 2 — Shadow run with live account**

```bash
python main.py --live-shadow --mode open
```

Verify `LIVE_SHADOW_COMPLETE` appears in the log and `would_buy_count > 0`. This confirms the AI is generating candidates and all gates are running against your live account state.

**Step 3 — Canary live run**

```bash
python main.py --mode open
```

Watch `logs/scheduler.log` (or the terminal) in real time.

**Step 4 — Verify the fill**

Within 60 seconds of the run completing, check:

```bash
python cli.py status
```

Confirm all four of these appear in `logs/investorbot.db`:

```sql
sqlite3 logs/investorbot.db \
  "SELECT event, payload FROM audit_events ORDER BY id DESC LIMIT 10;"
```

- `ORDER_TIMING` event: `fill_latency_ms` present, `stop_ok=True`
- `ORDER_EXEC_QUALITY` event: `fill_avg_price > 0`, `slippage_vs_mid_bps` present
- Trailing stop visible in Alpaca dashboard

**Step 5 — Manual close**

Close the canary position immediately. Do not hold it overnight.

```bash
# Force a sell via the open_sells mode (AI may decide HOLD; override by editing the position
# or using Alpaca dashboard for an immediate manual close).
python main.py --mode open_sells
```

**Step 6 — Pass / fail criteria**

| Check | Pass | Fail — stop and investigate |
|-------|------|------------------------------|
| Order placed | `FILLED` in audit log | `REJECTED` or no event |
| Fill price recorded | `fill_avg_price > 0` in `ORDER_EXEC_QUALITY` | `0.0` |
| Stop attached | `stop_ok=True` in `ORDER_TIMING` | `stop_ok=False` or stop missing |
| Slippage reasonable | `slippage_vs_mid_bps` between −50 and +50 | Outside ±100 bps |
| Position closed | Balance restored (minus spread/slippage) | Position still open |
| No HALT file | No `logs/.HALTED` file | HALT file written |

If all pass: canary complete. Proceed to [Going Live](#3-going-live).

If any fail: do not continue. Fix the root cause and repeat from Step 1.

---

## 3. Going Live

After canary passes, raise limits incrementally. Do not jump from canary sizing to full deployment in one step.

### 3.1 Suggested escalation

| Session | `MAX_SINGLE_ORDER_USD` | `MAX_POSITIONS` | Condition to advance |
|---------|----------------------|-----------------|----------------------|
| Canary | $20 | 1 | All canary pass/fail criteria met |
| Week 1 | $100 | 1 | No incidents, stops firing correctly |
| Week 2 | $250 | 2 | Fill quality consistent (avg slippage < 20 bps) |
| Week 3+ | Scale to plan | Up to 5 | No HALT events, health stays GREEN |

### 3.2 First live open run checklist

- [ ] `LIVE_CONFIRM=I-ACCEPT-REAL-MONEY-RISK` in env
- [ ] Safety check GREEN within 30 minutes of market open
- [ ] No macro risk events flagged (`python cli.py status`)
- [ ] VIX below 30 (above 30 → wider stops → more slippage → harder to manage small accounts)
- [ ] Terminal attached to watch logs in real time: `tmux attach -t investorbot`

---

## 4. Monitoring During Runs

### 4.1 Log locations

| File | What to watch |
|------|--------------|
| `logs/scheduler.log` | Run start/end, order confirmations, errors |
| `logs/investorbot.db` | Full audit trail — every order, event, decision |

### 4.2 Audit queries after each run

```sql
sqlite3 logs/investorbot.db

-- Last 20 events
SELECT ts, event, payload FROM audit_events ORDER BY id DESC LIMIT 20;

-- Execution quality for today's buys
SELECT ts, payload FROM audit_events
  WHERE event='ORDER_EXEC_QUALITY' ORDER BY id DESC LIMIT 5;

-- Any HALT or error events today
SELECT ts, event, payload FROM audit_events
  WHERE event LIKE '%HALT%' OR event LIKE '%FAIL%' ORDER BY id DESC LIMIT 10;
```

### 4.3 Health check between runs

```bash
python main.py --safety-check
```

Expected: GREEN. If YELLOW: investigate the flagged issue before the next run. If RED: do not run until resolved.

### 4.4 Stop coverage audit

Every position must have an active trailing stop. After each open run:

```bash
python main.py --safety-check
# Look for "Stop coverage: OK"
```

---

## 5. Incident Response

### 5.1 Stop placement failed after buy

Symptom: `ORDER_TIMING` shows `stop_ok=False`, or `STOP FAILED` alert received.

Response (live mode only):
1. Log in to Alpaca dashboard immediately
2. Manually set a stop-loss order for the unprotected position
3. Check if bot wrote a HALT file: `ls logs/.HALTED`
4. After manual stop placed: `python main.py --clear-halt && python main.py --safety-check`

### 5.2 Duplicate scheduler processes (lock race)

Symptom: `Lock file exists — another run may be in progress` repeated in `logs/scheduler.log`.

Response:
```bash
pgrep -af "python.*main.py\|python.*run_scheduler"
kill <stale_pid>
ls logs/.lock_*          # check for stale lock files
rm -f "logs/.lock_$(date +%Y-%m-%d)"   # remove only if > 30 minutes old
```

### 5.3 AI validation failure (structural)

Symptom: `AI response validation failed — blocking all Claude-driven decisions`.

Response: This is a safety feature. No orders were placed. Investigate the raw Claude response:

```sql
SELECT ts, payload FROM audit_events
  WHERE event='VALIDATION_FAILURE' ORDER BY id DESC LIMIT 3;
```

Review the LLM call log for the same run_id to understand what the model returned.

### 5.4 Order timeout (ambiguous fill)

Symptom: `ORDER AMBIGUOUS` alert, or `ORDER_TIMING` shows `status=TIMEOUT`.

Response:
1. Check Alpaca dashboard — did the order fill?
2. If filled at broker but not in DB: run `python main.py --safety-check` — `ensure_stops_attached` will detect and protect the position
3. If not filled: DAY order expired — no open exposure, no action needed
4. Resolve the stale intent manually if needed:
   ```sql
   UPDATE order_intents SET status='cancelled'
     WHERE status='timeout' AND symbol='XXXX';
   ```

---

## 6. Emergency Halt

Stop all trading immediately and liquidate all open positions:

```bash
python main.py --kill-switch
```

This cancels all open orders, submits market-sell for every position, waits for fill confirmation, and writes `logs/.HALTED`. The bot will refuse to run again until the halt is explicitly cleared.

To resume after the incident is resolved:

```bash
python main.py --clear-halt
python main.py --safety-check   # must be GREEN before resuming
```

**Use the kill switch only when you need positions closed now** — e.g., unexpected account drawdown, repeated stop failures, broker connectivity lost. It is not a debugging tool.
