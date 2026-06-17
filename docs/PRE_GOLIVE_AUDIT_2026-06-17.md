# Pre-Go-Live Audit — 2026-06-17

A ten-part sweep before the live experiment, following the integrity audit (1.104) and the
sector-cache finding (F7 / 1.105-1.106). Tier 1 (risk, lookahead, parity) got the deepest
investigation; Tier 2 moderate; Tier 3 a lighter pass — items flagged `[needs deeper review]`
were sampled, not exhaustively traced.

## Meta-findings (the patterns, not the instances)

1. **Silent graceful degradation** — the recurring root cause of *every* issue found to date. The
   bot defaults to a safe-*looking* fallback (Unknown sector, empty cache, clobbered baseline,
   stale-but-"today" data, fail-open gates) instead of surfacing the failure, so problems
   accumulate invisibly. Defensive coding that hides defects is the single biggest systemic risk.
2. **Live-only filters not in the backtest (parity drift)** — a *new* pattern surfaced here: the
   live pipeline applies selection filters the validated backtest never had, so live results don't
   reflect the validated edge. F7 made one of these (the sector gate) go from accidentally-inert to
   actively divergent.

---

## Tier 1 — before go-live (capital + experiment validity)

### Audit 1 — Risk-controls firing

**Positive:** the core enforcement is sound. `skip_buys` (main.py:2101) correctly blocks new buys
on circuit breaker, bearish regime, macro event, health RED/YELLOW, and experiment-drawdown; the
daily-loss path liquidates all positions and halts on any close failure; the 15% notional cap
(`MAX_POSITION_PCT`) and max-positions slots enforce; F1 fixed the daily-loss/breaker baseline.

- **A1.1 [HIGH — fixed by F7]** The sector **concentration cap** was *also* silently disabled by the
  empty sector cache. `validate_buy_candidates` (risk_manager.py:102) exempts `"Unknown"` sectors —
  so pre-F7, with ~every symbol "Unknown", the bot had **no sector-diversification enforcement** and
  could pile into one sector. F7 (populating the cache) restored it; the momentum gate was not the
  only beneficiary.
- **A1.2 [MED]** The **circuit-breaker lookback is record-count-based**, not calendar days:
  `portfolio_history[-5:]` (risk_manager.py:19). With 4 runs/day that's ~1.25 days, not the
  documented "5-day peak." Recommend: collapse to one record/day (e.g. the close run) or use a
  time-window before taking the peak.

### Audit 2 — Lookahead / point-in-time

**Positive:** the experiment backfill/dataset machinery is lookahead-clean *by design* —
`AsOfExpectancy` queries only `known_at < decision_date` (strict, dataset.py:83), entry indexed at
the decision date (backfill.py:25), forward returns are the measured outcome (not a decision input),
and it's documented as such.

- **A2.1 [MED — needs deeper review]** Residual risk is the **live feature pipeline**, not the
  experiment code: do fundamental/earnings features use *point-in-time* (as-reported) dates rather
  than latest-available? A single as-of leak in the live features (e.g. using a restated fundamental
  or a not-yet-announced earnings figure) would bias selection. Not verified this pass — the highest
  remaining lookahead risk.

### Audit 3 — Live↔backtest parity  ·  **headline finding**

- **A3.1 [HIGH]** The **sector-momentum long gate** (main.py:2217) and the **sector concentration
  cap** are **live-only — the backtest applies neither** (no `get_sector` / `sector_allowed_long` /
  `sector_momentum` anywhere in `backtest/engine.py`). The **churn guard (F4)** is also live-only.
  Consequence: pre-F7 the gate was inert live (Unknown → fail-open), so live ≈ backtest *by
  accident*; **F7 activated it, so live now diverges from the validated backtest.** Our own fix
  created a parity gap. This directly bears on "why does live differ from the backtested edge."
  **Decision required:** either add these filters to the backtest and re-validate the edge with them
  on, or consciously decide whether to apply unvalidated filters in live trading. (Supersedes the
  earlier "confirm backtest sector-gate parity" item — answer: it does not.)

---

## Tier 2 — high value, soon

### Audit 4 — Execution / fills
**Positive:** slippage/spread are modeled in the **backtest only** (correct — live gets real fills);
idempotency is enforced via the order-intent ledger, which hard-blocks new buys in live if the
ledger is unavailable (trader.py:126); partial fills are handled (filled_qty / late_qty).
- **A4.1 [LOW]** **Dust accumulation** — fractional residual positions (e.g. CI at qty 7.89e-07)
  linger with no automatic sweep until the AI happens to notice and "clean up." Recommend a
  dust-threshold auto-close.

### Audit 5 — Fail-safe / resilience
- **A5.1 [MED-HIGH — needs deeper review]** This is the meta-theme made into work: graceful
  degradation is pervasive and *hides* failures. A systematic pass should enumerate **every external
  dependency** (market data, broker, LLM, each feed) and **every degradation path**, and confirm
  each one (a) surfaces visibly in logs/alerts and (b) fails **safe** (skip trading) not **open**
  (trade on bad/default data). Known fail-open instances found so far: the sector gate (pre-F7) and
  the data feeds (pre-1.104). Not exhaustively audited.

### Audit 6 — Paper→live safety
**Positive / low concern:** `config.validate()` hard-fails on any `TRADING_MODE`↔`ALPACA_BASE_URL`
mismatch (config.py:50-65), the broker client is constructed `paper=IS_PAPER`, and the idempotency
ledger hard-blocks only in live. Accidental real-money trading via config drift is well-guarded.

---

## Tier 3 — lighter pass, flagged for deeper work

### Audit 7 — Experiment power / design
- **A7.1 [MED-HIGH — acknowledged in design]** Per `docs/EXPERIMENT.md` and the daily report, the
  live experiment is **underpowered for its primary IC endpoint** ("projected underpowered for the
  live track; scoped as a trend and qualitative layer"), and **Gate A (noise audit) has not been
  run.** So the live arm cannot, alone, rigorously test H1 — it leans on `N_eff` accumulation
  (interim ≥200, primary ≥400) plus qualitative case studies. Worth a clear-eyed decision: can the
  experiment as-designed answer its question, or does it need a longer horizon / a backtest-anchored
  arm? This is the project's *purpose*, so it ranks higher than its tier.

### Audit 8 — Timezone / DST
- **A8.1 [MED — latent]** **Three timezone conventions coexist:** ET (scheduler, `today_et`), UTC
  (the daily P&L baseline date), and local/UK (`date.today()` for regime cache keys at
  market_regime.py:669/691/695/763/777; log timestamps). It works *during US market hours* (UK date
  == ET date then) but is fragile at day boundaries and across DST shifts. Also
  `datetime.utcnow()` (market_regime.py:621) is deprecated + naive. Recommend standardizing all
  trading-day logic on `today_et()`.

### Audit 9 — Reconciliation / state consistency
- **A9.1 [LOW-MED — needs deeper review]** Position metadata (DB) vs broker truth is reconciled at
  startup (`reconcile_positions` + the "unexpected positions" halt), which looks reasonable. Dust
  (A4.1) is the main hygiene gap. Cache coherence across the 4 daily runs + the prefetch thread (the
  `build_sector_map` race we hardened in 1.106) is partially addressed. Not exhaustively traced.

### Audit 10 — Cost-model realism
- **A10.1 [MED — open]** The backtest models costs (5bps slippage + ADV-scaled spread + impact);
  live realizes actual fills. Whether modeled costs **match realized** live fills is unverified —
  compare decision price vs actual fill price across accumulated paper trades. If modeled < realized,
  the validated edge overstates live performance.

---

## Priority for action

1. **A3.1 (parity)** — decide on the live-only filters; it explains live/backtest divergence and
   gates any strategy conclusion.
2. **A2.1 (live-pipeline lookahead)** — the one remaining place a leak would invalidate the
   experiment.
3. **A5.1 (resilience sweep)** — systematize the fail-safe verification; it's the root pattern.
4. **A7.1 (experiment power)** — a clear-eyed decision on whether the live experiment can answer H1.
5. **A1.2, A8.1, A4.1, A10.1, A9.1** — fix as a batch of hardening.

Confirmed-clean / positive: core risk enforcement (A1), experiment backfill machinery (A2),
paper→live guard (A6), execution idempotency + cost separation (A4).

---

## Resolution log

- **A1.2 — FIXED (1.107):** circuit-breaker lookback now per-calendar-day (5 days), not per-record.
- **A4.1 — FIXED (1.107):** dust sweep auto-closes sub-$1 fractional residuals in the sell phase.
- **A8.1 — PARTIAL (1.107):** `datetime.utcnow()`→`datetime.now(UTC)` fixed; the latent
  `date.today()`→`today_et()` cache-key standardization deferred (never triggers given the run
  schedule; disproportionate test-churn/risk for a theoretical issue).
- **A2.1 — VERIFIED CLEAN (no fix):** lookahead controls are strong and explicit (backtest signals
  point-in-time-safe "strictly before sim_date"; backfill `known_at < decision_date`; live is
  real-time). Sole known leak (distress-short fundamentals, pre-2020) is documented + scoped.
- **A1.1 — already fixed by F7 (1.105):** sector concentration cap restored (was Unknown-exempt).
- **A3.1, A7.1 — OPEN (decisions, not mechanical fixes):** parity (add live-only filters to backtest
  + re-validate vs remove from live); experiment power (can it answer H1 as designed).
- **A5.1, A9.1, A10.1 — OPEN (investigations / need data):** resilience fail-safe sweep;
  reconciliation depth; modeled-vs-realized cost (needs accumulated paper trades).
