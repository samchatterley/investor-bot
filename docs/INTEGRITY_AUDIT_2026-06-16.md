# Data Integrity Audit — 2026-06-16

Triggered by a mismatch between the trading-summary email (−$819.73) and the logged
`daily_pnl` (−$310.27) for 2026-06-16. The audit confirms the **email is correct** and
several **internal logs/fields and one risk gate are not**. None of the issues are a
signal-book failure — they are instrument (data/accounting) defects.

## Executive summary

| | Value | Source |
|---|---|---|
| **True daily P&L** | **−$819.73 (−0.83%)** | open-of-day equity $99,029.49 → close $98,209.76 |
| Logged `daily_pnl` (close.json) | −$310.27 | wrong — measured from a clobbered baseline |
| SPY today (Jun 16) | **−0.60%** | live yfinance (754.83 → 750.33) |
| Bot's reported market | "+1.7% bull trend" | wrong — that is Jun 15's move, reused all day |

The day's loss is explained by: **stale market read → regime misclassified as BULL_TREND →
bot deployed ~$57k (20%→61% invested) into a falling market → intraday churn.** The
scoreboard then under-reported the damage because the P&L baseline was clobbered.

## Trustworthy vs not

- **Trustworthy:** the email (uses true market-open baseline); `logs/scheduler.log` order
  fills (ground-truth broker confirmations); live prices.
- **Not trustworthy:** `daily_pnl` in the per-run JSONs; the `market_summary` narrative and
  regime label; the daily-loss circuit breaker's loss figure.

---

## Findings

### F1 — `daily_pnl` measured from the wrong, mutable baseline · **HIGH (reporting + risk)**

`save_daily_baseline()` is called **only on `mode == "open"`** (`main.py:3053`) using *that
run's* `account_before`, and it **overwrites** `logs/daily_baseline.json` every call
(`utils/portfolio_tracker.py:260`, `open(..., "w")`).

- The true market open is the **open_sells** run (09:31 ET), equity **$99,029.49**.
- The baseline is instead set at the later **open** run (10:00 ET) = **$98,826.73**
  (already ~$200 below the true open — market moved in between).
- `daily_baseline.json` currently holds **$98,520.03**, an *afternoon* value. Because the
  save fires only on `mode=open`, an **extra open-mode invocation after midday clobbered it**
  (a `python main.py` restart starts open mode — the exact thing `CLAUDE.md` warns against).

Result: `close.json daily_pnl = 98,209.76 − 98,520.03 = −310.27`, vs the true
`98,209.76 − 99,029.49 = −819.73`. The email uses the true open, so it is correct; the JSON
uses the clobbered baseline, so it is wrong. (`daily_pnl` is then recomputed at
`portfolio_tracker.py:122–124` from this baseline, overriding the per-run delta.)

**Risk impact:** the daily-loss circuit breaker uses the *same* baseline
(`main.py:2916` `load_daily_baseline()`, `main.py:2922` `daily_loss_usd = _baseline −
account_before`, `risk/risk_manager.py:47` `check_daily_loss(value_at_open, …)`). A baseline
clobbered *low* understates the daily loss → **the breaker can fail to trip** when real
losses exceed the limit.

**Fix:** set the baseline exactly once per trading day, at the first run (open_sells), from
the true market-open equity; never overwrite within the day; make a `mode=open` restart not
re-stamp an existing same-day baseline.

### F2 — Stale market narrative drives live decisions · **HIGH (decision quality)**

Both `2026-06-16.json` (open, 10:03 ET) and `2026-06-16-close.json` (15:33 ET) say
*"Market in confirmed bull trend with SPY +1.7% today driven by US-Iran peace deal …"*.
Live reality: **Jun 16 SPY −0.60%, VIX 16.4.** The "+1.7% / VIX 16.1" matches **Jun 15**
(741.75 → 754.83 = +1.76%, VIX 16.20) exactly. The regime label flipped **NEUTRAL_CHOP
(Jun 15) → BULL_TREND (Jun 16) on a down day.**

**Impact:** the bot went from ~20% invested ($19.6k) to ~61% ($60k), buying 5 names into a
−0.6% session it believed was a +1.7% bull. New positions were underwater by the close
(KEYS −1.9%, RL −1.4%).

**Root cause (hypothesis):** "today's" move/regime is computed from the last *complete* daily
bar, which is yesterday's until after the 16:00 ET close — then labelled "today." Same family
as the stale-feed idle week (see F6).

**Fix:** compute the current-day move from intraday data, or label it as prior-session and do
not flip the regime on it; assert price freshness at every run.

### F3 — Inconsistent run-file naming; `date` param overloaded with mode · **MEDIUM (observability)**

`_daily_log_path()` returns `{date}.json` with **no mode suffix**
(`utils/portfolio_tracker.py:36`). The `mode=open` run passes a plain date and writes the
**bare `2026-06-16.json`**; open_sells/midday/close pass a mode-suffixed *date* string
(`"2026-06-16-midday"`, see the mode parsing at `portfolio_tracker.py:133–137`) and so write
`2026-06-16-{mode}.json`. The bare file is easy to miss — it caused this audit's first pass to
wrongly conclude the KEYS/FCX/HPE buys were unlogged. **They are logged**, in
`2026-06-16.json`. (Flagged by the user.)

**Fix:** name every run file `{date}-{mode}.json` (open → `2026-06-16-open.json`); stop
overloading the `date` parameter to carry the mode; reconcile the mode name with the
documented "open_buys".

### F4 — Intraday churn / same-day round-trips · **MEDIUM (cost + coherence)**

HPE was **bought at 10:03 ET** ($6,106, conf 7/10 — "guidance raise, unusual call activity,
bullish hammer") and **sold ~2 hours later at 12:03 ET** (−$180 — "Bear of the Day, RSI 90.2,
MACD rolling over"). The fractional CI position was also churned. Same-name round-trips in one
session pay the spread twice and reveal a self-contradictory conviction (likely the same
stale-vs-intraday data problem from F2).

**Fix:** min-hold guard / block same-day exits absent a stop trigger; root-cause why the model
flips on one name within hours.

### F5 — Per-run account snapshots do not chain · **LOW (verify; may be benign)**

`open_sells.json` after `cash=$79,402.18`, but `open.json` before `cash=$88,939.51`
(+$9,537 with no logged trade in the gap); `pv` 99,037 → 98,826. Possibly normal broker
settlement/buying-power timing, but each run is an independent broker fetch — worth confirming
that `before == prior after ± market move`.

### F6 — Stale-feed health-gate idle week · **HIGH (previously diagnosed)**

Early–mid June: stale yfinance feed → startup health YELLOW/RED → new buys suspended → ~a week
all-cash through a +2% SPY move. Same data-freshness root cause as F2. See memory
`project_pnl_was_feed_outage`.

---

## Priority

1. **F1** — P&L baseline + daily-loss breaker (one fix repairs both reporting and a risk gate).
2. **F2** — market-data freshness (drives bad buys; same root as F6).
3. **F4** — churn control.
4. **F3** — run-file naming (quick; the user's catch).
5. **F5** — snapshot chaining (investigate; may be benign).

**Strategy work (signal book, beat-SPY, experiment Arms) stays deferred until F1–F4 are fixed
— you cannot evaluate or rebuild a strategy on misreported P&L and day-stale prices.**
