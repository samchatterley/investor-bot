# InvestorBot Live Runbook

Exact procedures for operating the bot in live mode.
Run `python main.py --safety-check` before any live session.

---

## Pre-Market Checklist (run every live trading day before 09:00 ET)

```
[ ] python main.py --safety-check          → must exit 0 (GREEN)
[ ] Confirm Alpaca account balance matches expected capital
[ ] Confirm no unexpected positions at broker
[ ] Confirm no stale open orders at broker (Alpaca dashboard)
[ ] Confirm scheduler is running: pgrep -a caffeinate | grep run_scheduler
[ ] Confirm log rotation / disk space: df -h logs/
[ ] Confirm .env has correct ALPACA_BASE_URL for today's mode (paper/live)
[ ] Confirm SMALL_ACCOUNT_MODE=true if running £150 experiment
[ ] Confirm LIVE_CONFIRM=I-ACCEPT-REAL-MONEY-RISK is set in .env (live only)
[ ] Confirm no HALT file: ls logs/.HALTED 2>/dev/null && echo HALTED || echo OK
```

---

## First Live Trade Checklist (one-time, before first real-money order)

```
[ ] Run python main.py --live-shadow --mode open  → WOULD_BUY decisions only, no orders
[ ] Verify --live-shadow pulls correct live account balance
[ ] Verify --live-shadow shows correct open positions (should be none)
[ ] Verify --live-shadow runs all risk gates (pre-trade, exposure cap, etc.)
[ ] Confirm broker-side: margin disabled, PDT flag absent, no options access
[ ] Confirm dedicated Alpaca account holds only experiment capital (≤ £150)
[ ] Confirm read-only key is separate from trading key
[ ] Complete canary run (see CANARY section below)
[ ] Re-run --safety-check after canary — must still be GREEN
```

---

## What To Do On: Halt File Present

The bot created `logs/.HALTED` and stopped accepting new orders.

```bash
# 1. Read why it halted
cat logs/.HALTED | python3 -m json.tool

# 2. Check broker state
python main.py --safety-check

# 3. If positions are unprotected — attach stops manually in Alpaca dashboard
#    or via CLI (replace SYMBOL/QTY/STOP_PRICE):
#    python cli.py place-stop SYMBOL QTY STOP_PRICE

# 4. If position is still open and stop cannot be attached — close it:
#    python cli.py flatten SYMBOL

# 5. Once broker state is clean:
python main.py --clear-halt

# 6. Re-run safety check to confirm GREEN before resuming scheduler
python main.py --safety-check
```

---

## What To Do On: Missing Stop

Symptom: `--safety-check` reports "no stop protection: SYMBOL".

```bash
# 1. Check what's at broker
python main.py --safety-check

# 2. Run ensure_stops_attached via safety check (it calls it automatically)
#    If the stop was just delayed, --safety-check will attach it.

# 3. If stop still cannot be placed (illiquid, halted symbol):
python cli.py flatten SYMBOL   # close the position
python main.py --safety-check  # confirm clean

# 4. If symbol is halted at broker and cannot be closed:
#    - Contact Alpaca support
#    - Do NOT resume the bot until the position is resolved
#    - Write halt file manually if needed: python main.py --kill-switch
```

---

## What To Do On: Alpaca API Outage

Symptom: Bot logs "BrokerStateUnavailable" and suspends buys.

```bash
# 1. Check Alpaca status: https://status.alpaca.markets/
# 2. Do nothing — the bot suspends buys on broker uncertainty (fail-closed design).
# 3. Existing stops remain at broker and continue to protect positions.
# 4. When Alpaca recovers, the next scheduled run will proceed normally.
# 5. If outage was during an active order (status SUBMITTED in ledger):
python main.py --safety-check   # check for unresolved intents
# Resolve any 'submitted/timeout' intents by checking Alpaca dashboard manually.
```

---

## What To Do On: Duplicate Order Rejection (409)

Symptom: Buy order returns REJECTED with "client_order_id already exists".

This is expected and safe behaviour — it means the symbol+date client_order_id
was already submitted today (likely from a prior run). The order ledger will
show status=rejected for the duplicate attempt.

```bash
# 1. Confirm the original order exists at broker (Alpaca dashboard).
# 2. If the original filled: position is protected, nothing to do.
# 3. If the original was rejected/cancelled: the 409 is a false positive.
#    Delete the order_intents ledger entry for today and rerun:
#    python cli.py clear-intent SYMBOL  (or manually via SQLite)
```

---

## What To Do On: Partial Fill

Symptom: Buy order returns PARTIAL status.

```bash
# 1. --safety-check will detect the position and check stop coverage.
# 2. The filled quantity is recorded; a stop is placed for whole shares.
# 3. If filled qty < 1 whole share: position is UNPROTECTED (Alpaca limitation).
#    Close the fractional position immediately:
python cli.py flatten SYMBOL
```

---

## How To Manually Flatten a Position

```bash
# Option 1: CLI helper
python cli.py flatten SYMBOL

# Option 2: Alpaca dashboard — Markets → Positions → Close

# Option 3: API direct (emergency)
python -c "
from execution.trader import get_client, close_position
c = get_client()
r = close_position(c, 'SYMBOL')
print(r)
"
```

---

## How To Resume Safely After Halt

```bash
# 1. Understand why it halted (read HALT file)
cat logs/.HALTED | python3 -m json.tool

# 2. Fix the underlying condition (stop, position, cap breach)

# 3. Run safety check — must be GREEN before resuming
python main.py --safety-check

# 4. Clear halt
python main.py --clear-halt

# 5. Confirm scheduler is running; if not, restart it:
pgrep -a caffeinate | grep run_scheduler || \
  nohup caffeinate -i .venv/bin/python scripts/run_scheduler.py >> logs/scheduler.log 2>&1 &
```

---

## What Logs To Preserve After An Incident

```bash
# Copy these before clearing anything:
cp logs/.HALTED      incident_$(date +%Y%m%d)/halt.json
cp logs/audit.jsonl  incident_$(date +%Y%m%d)/audit.jsonl
cp logs/scheduler.log incident_$(date +%Y%m%d)/scheduler.log
sqlite3 logs/investorbot.db ".dump order_intents" > incident_$(date +%Y%m%d)/order_intents.sql
sqlite3 logs/investorbot.db ".dump order_events"  > incident_$(date +%Y%m%d)/order_events.sql
```

---

## Canary Procedure (Run Before Full Experiment)

A controlled single-symbol test with a tiny cap to confirm the full
buy → fill → stop pipeline works end-to-end.

```bash
# 1. Set canary config (copy .env.canary → .env temporarily, or export):
export MAX_SINGLE_ORDER_USD=10
export MAX_DEPLOYED_USD=10
export MAX_DAILY_NOTIONAL_USD=10
export MAX_ORDERS_PER_RUN=1
export MAX_POSITIONS=1

# 2. Run safety check
python main.py --safety-check   # must be GREEN

# 3. Run live-shadow to confirm sizing
python main.py --live-shadow --mode open

# 4. Place the canary order (highly liquid symbol, price < $10)
python main.py --mode open

# 5. Confirm after run:
[ ] Buy submitted once (check logs/audit.jsonl for ORDER_PLACED)
[ ] No duplicate on rerun (run again → INTENT_ACTIVE log, no second order)
[ ] Stop attached (check Alpaca dashboard → Orders → stop order present)
[ ] Stop visible at broker
[ ] Local DB matches broker: python main.py --safety-check
[ ] Kill switch works: python main.py --kill-switch
[ ] Position closed and HALT written
[ ] Resume works: python main.py --clear-halt && python main.py --safety-check

# 6. If all checks pass: raise caps to full £150 experiment profile (.env)
```

See `.env.canary` for the full canary environment template.

---

## Incident Drills

Run these in paper mode to verify safety mechanisms work before going live.

```bash
make drill-stop-failure        # simulate stop placement failure → expect flatten + alert
make drill-broker-timeout      # simulate get_orders timeout → expect BrokerStateUnavailable
make drill-restart-mid-order   # simulate crash between submit and fill → expect intent reconciliation
make drill-kill-switch         # activate kill switch → expect halt file + positions closed
make drill-missing-stop        # start with unprotected position → expect ensure_stops_attached
make drill-duplicate-order     # submit same client_order_id twice → expect 409 REJECTED, no double-buy
```

---

## Escalation

If any of the above procedures fail and the bot has a live unprotected position
you cannot close programmatically:

1. Log into Alpaca dashboard immediately
2. Close position manually
3. Revoke API keys (Settings → API Keys → Delete)
4. Contact Alpaca support if position cannot be closed: support@alpaca.markets
