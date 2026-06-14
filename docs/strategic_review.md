# Strategic & Architectural Review — InvestorBotHard

> Companion to `docs/audit_v1.100.md`. That document asks "is the code implemented
> correctly?" (answer: largely yes). This one asks "is this the right machine to be
> building at all?" It is a senior-architect / quant-research-lead / product-strategy
> review, not a bug hunt.

A different lens from the line-by-line audit. Grounded in: the LLM boundary
(`analysis/ai_analyst.py`), the live buy gate (`main.py:2040-2128`), the feedback loop
(`analysis/weekly_review.py`), the backtest's signal path (`backtest/engine.py:61-72`),
and the capital config (`config.py:69-118`).

**Verdict up front:** the bolts are tight and several core decisions are genuinely sound,
but the machine has an identity problem that distorts almost everything downstream.

---

## The central tension: a research platform wearing a trading-bot costume

The single most important fact in the repo is `config.py:69`: `SMALL_ACCOUNT_MODE` is a
**£150-scale live run**. Around that £150 sits ~33,000 lines: an options IV surface, EDGAR
13D parsing, a borrow-cost model, FRED macro calendars, Google Trends, sentiment, an index
regime hedge, and a per-trading-day Claude API call.

**Classification: Design pivot candidate (the framing one).**

- **Current approach:** Build a near-institutional decision stack and point it at a
  retail-experiment account, treating P&L as the objective.
- **Why it's a problem:** At £150 with 1–2 positions (`config.py:77`) and 1 order/run
  (`config.py:810`), three things are mathematically true:
  1. **You will never reach statistical significance on live trade outcomes** — at a
     handful of trades a month, per-signal win rates need *years* to separate from noise,
     yet `empirical_win_rate` already feeds sizing.
  2. **Frictions dominate any edge** — spread, slippage, and borrow on sub-£10 fractional
     positions swamp a +0.16% backtested signal avg.
  3. **The LLM call costs real money to decide on a position worth pennies** — the
     economics only make sense if the *output is learning, not profit*.
- **Deeper concern:** Every design argument ("should we add factor risk?", "is the short
  book worth it?") is unanswerable until you decide whether the objective is *P&L on this
  account* (it can't be — the account is too small to matter) or *building and validating a
  process that could later run real size* (almost certainly the real goal). The code
  optimizes per-trade decisions as if for P&L, but the account says research.
- **Better approach:** Declare the identity explicitly. If it's a **research /
  decision-support platform**, then the primary output artifact is *evidence about signals
  and process*, the live £150 is a realism harness, and success is measured in validated
  edges and calibration — not daily P&L. That reframing changes what you build next.
- **Urgency: Essential now** — a free decision that re-prioritizes everything else.

---

## 1. Problem framing & objective function

### 1.1 — There is no portfolio-level objective. *(Strategic weakness)*

The system optimizes a *sequence of individual trade decisions*: signal fires → LLM scores
confidence → gate → per-name sizing with ~10 scalars (`main.py:2117-2128`). Nothing
optimizes the *portfolio* as an object. `correlation_scalar` and `MAX_SECTOR_POSITIONS` are
per-name heuristics, not a covariance-aware construction. There is no target Sharpe, no
drawdown-budgeted return, no factor-exposure view. The weekly review optimizes *narrative
lessons*, not an objective.

- **Deeper concern:** A book of 2–5 names each picked for an idiosyncratic signal can still
  be one concentrated bet (all long low-quality momentum into a CPI print) without any
  component noticing — because no component reasons about the book as a whole.
- **Better approach:** Define one objective function (risk-adjusted return subject to a
  drawdown budget) and make sizing/selection serve it. Even a simple portfolio layer — net
  exposure, factor tilt, correlation of the *proposed book* — beats 10 independent per-name
  scalars.
- **Urgency: Important later** (essential at scale).

---

## 2. Architecture & the LLM's placement

### 2.1 — The LLM is well-contained, but it gates on the system's *weakest* evidence. *(Strategic weakness — the important one)*

Credit first: the LLM placement is disciplined. It cannot invent candidates (they must pass
the deterministic prefilter), cannot size (explicitly forbidden, `ai_analyst.py:428`),
cannot bypass risk gates, and its cited `key_signal` is cross-checked against
actually-fired signals to catch hallucination (`main.py:2105-2115`). Output is a constrained
tool schema with domain validation and a full audit trail. This is how to use an LLM safely.

**But** look at what the LLM actually decides: a 1–10 **confidence integer that is the entry
gate** (`confidence >= min_confidence`, `main.py:2069`). Meanwhile the system's *best*
evidence — empirical per-signal/per-regime win rates — only flows into *sizing* conviction,
not the gate. So:

> The most consequential decision (trade / don't trade) is made by the least
> evidence-grounded number in the system (an LLM's subjective confidence), while the most
> evidence-grounded number (empirical edge) merely scales size.

- **Deeper concern:** LLMs are poorly calibrated probability estimators and will happily
  produce a confident 8/10 narrative for a coin flip. You've put the narrative in charge of
  the gate and the data in charge of the dial. That's inverted.
- **Better approach:** Gate on **evidence** (does this signal, in this regime, have a
  statistically credible edge?) and demote the LLM to what it's actually good at:
  **synthesis and veto** — reading news/filings/options context to *suppress* a trade the
  deterministic edge would otherwise take ("earnings tomorrow," "the 13D is a financing
  arm, not an activist"). Let the LLM lower confidence, not be the source of it.
- **Urgency: Important now** — the highest-leverage conceptual change in the repo.

### 2.2 — `backtest/engine.py` is a 6,769-line monolith; `main.py` is 3,232. *(Architectural bottleneck)*

20% of the source is one file. The signal *evaluator* is correctly shared (see 5.1), but
the engine bundles data prep, simulation, costs, short logic, intraday, and reporting.

- **Deeper concern:** Walk-forward validation, a second strategy, or a second broker all
  require surgery on a god-file with high blast radius. This is the component most likely to
  break first under extension.
- **Better approach:** Split into `engine/` (loop), `costs/`, `portfolio/`, `reporting/`;
  make the simulation step a thin driver over the same `evaluate_signals` + a `Portfolio`
  object the live path also uses.
- **Urgency: Important later.**

### 2.3 — Execution and data are hard-coupled to Alpaca and yfinance. *(Architectural bottleneck)*

Alpaca order types thread through `execution/trader.py`; yfinance is the spine of the data
layer (rate-limited, unofficial, single-source). Fine for research; a rewrite-magnet the
moment you add a broker or need reliable quotes. Introduce a thin `Broker` and `MarketData`
port now while it's cheap. **Urgency: Important later.**

---

## 3. Research validity — the biggest gap

### 3.1 — You backtest one machine and run a different one. *(Research/process gap — the critical one)*

The backtest delegates detection to the canonical `evaluate_signals` (excellent — no signal
divergence). But the backtest trades **every signal that fires** through the deterministic
gate. Live, the **LLM re-ranks and selects a subset** by confidence. The LLM is *not in the
backtest* (it can't be — non-deterministic). And `live_shadow` is paper-alongside-live,
**not** a deterministic-vs-LLM counterfactual.

> Therefore: live performance = (backtested deterministic edge) ⊕ (LLM selection effect),
> and **the second term is never measured.** You cannot currently answer "is the LLM helping
> or hurting?"

- **Better approach:** Log the **counterfactual** on every run — what the deterministic-only
  system *would* have picked vs. what the LLM picked — and track the two equity curves in
  parallel (the shadow infra is 90% there). After N trades you have an evidence-based answer
  on whether the LLM earns its place. The single most valuable instrument you could add.
- **Urgency: Essential now** (cheap, and it gates the LLM-placement decision in 2.1).

### 3.2 — Signal research is narrative-driven, not a disciplined pipeline. *(Research/process gap)*

Signals are added in *batches* (v1.97 shipped 21 at once) and pruned ad-hoc when they
"drag," using elimination scripts on **in-sample** windows (2020–2022, 2015–2026). The
holdout guard *warns, doesn't block* (the B-obs finding). There are 41 long + 23 short
signals.

- **Deeper concern:** A textbook garden-of-forking-paths. With 64 signals and in-sample
  tuning, some will look good by chance; the £150 account will never generate enough live
  trades to demote them on evidence. You're accumulating *convincing narratives*, not
  *robust alpha*. The disable decisions shipped recently (OBV, the lagging shorts) are
  reasonable, but they're decided the same in-sample way the signals were added.
- **Better approach:** A signal lifecycle with an **evidentiary burden** at each gate:
  *discovery* (in-sample) → *validation* (walk-forward, out-of-sample, multiple-testing-aware
  threshold) → *paper/shadow* (live data, no capital) → *promotion* (only on out-of-sample
  edge that survives a deflated-Sharpe / Bonferroni-style haircut) → *monitoring* (auto-flag
  decay). Make the holdout **block, not warn**. Treat every new signal as guilty until
  proven.
- **Urgency: Essential now** (the difference between research and theatre).

---

## 4. Iterative improvement / feedback loop

### 4.1 — The learning loop is "soft" and open-ended. *(Research/process gap, with a strength inside it)*

The loop is: trades → DB attribution → Claude writes *probabilistic text lessons* injected
into next week's prompt for 7 days → expire (`weekly_review.py`). Config auto-changes are
deliberately **validated-but-never-applied** to prevent drift (`weekly_review.py:68-73`) —
that restraint is *correct and should be preserved*.

- **Deeper concern:** The only thing that actually adapts week-to-week is *prose in a
  prompt*, derived from ~a week of tiny-sample trades, fed back into the same LLM. That's a
  narrative echo, not a measured controller. There's no closed loop from out-of-sample
  outcomes → signal weights → deployment with a statistical gate. The system cannot
  *systematically* detect signal decay or regime failure; it can only have Claude muse about
  it.
- **Better approach:** Replace (or supplement) text-lessons with a measured controller:
  per-signal rolling out-of-sample expectancy with confidence intervals; auto-flag when a
  live/shadow signal's CI drops below its backtested edge (decay detection); feed *that* into
  the evidence gate from 3.2. Keep the human-in-the-loop for promotion/demotion — but on
  statistics, not vibes.
- **Urgency: Important now.**

---

## 5. Strengths that are strategically sound — do not change these

Real and load-bearing. Preserve them deliberately.

- **5.1 — Single canonical signal evaluator shared by backtest and live**
  (`backtest/engine.py:61-72` → `signals.evaluator`). The hardest thing to get right in a
  trading system, and it's right. *Keep it inviolable.* (Strategic strength.)
- **5.2 — Fail-closed execution core** (BrokerStateUnavailable / ledger-unavailable → block;
  crash-safe order ledger; account-relative drawdown circuit breaker). Correct
  risk-of-ruin posture. (Strategic strength.)
- **5.3 — Dormant-by-design self-modification** — nothing writes `runtime_config.json`; the
  auto-tuner proposes but never applies. Exactly the right call for an LLM-in-the-loop
  system. (Strategic strength.)
- **5.4 — Structured-tool LLM output + independent domain validation + audit trail +
  hallucination cross-check.** The reference pattern for safe LLM use. (Strategic strength.)
- **5.5 — Deterministic sizing with Kelly as telemetry-only.** Conviction comes from data,
  not the LLM, and the dangerous knob (Kelly) is explicitly barred from live notional.
  (Strategic strength.)

---

## 6. Risk philosophy & 7. Operational design (brief — mostly sound)

Per-trade risk is conceptually right (fat-finger, daily-notional, drawdown breaker, regime
gates, VIX-adjusted stops, fail-closed data). The **gaps are conceptual, not
implementational**: risk is *reactive and per-position*, with **no portfolio construction**
(4.1, 1.1) and **no factor/covariance view** — the index hedge is bolted on, not *netted*
against the long book's actual beta. Operationally the system handles missing data / partial
fills / broker errors well for a single-broker single-account research rig; it is **not
yet** shaped for live capital at size (single data source, serial per-symbol fetch before
the buy window, Alpaca lock-in).

---

## Final verdict

**1. Is the approach directionally correct?**
**Yes, as a research/decision-support platform; no, as a P&L-seeking trading bot.** The
engineering discipline is excellent and the safety architecture is better than most
production systems. But the project is mis-framed as a trader when its account, cadence, and
per-decision cost only make sense as a research harness. Re-label it and it's directionally
*right*; leave it mislabeled and you'll keep optimizing a number (£150 P&L) that cannot
teach you anything.

**2. Biggest strategic risk:** Identity confusion — pouring institutional engineering into a
P&L objective the account can never satisfy, so effort compounds on the wrong metric.

**3. Biggest architectural risk:** The 6,769-line backtest monolith + Alpaca/yfinance
coupling — the first things to break when you try walk-forward validation, a second
strategy, or real data.

**4. Biggest research-validity risk:** You validate the deterministic pipeline but run an
LLM-selected subset, and never measure the LLM's marginal effect — compounded by in-sample,
multiple-testing-blind signal proliferation behind a holdout that only warns.

**5. Biggest live-trading risk:** No portfolio-level view — a 2–5 name book can be one
concentrated, correlated, factor-loaded bet into a known macro event, and nothing in the
system reasons about the book as a whole.

**6. Top 5 design changes (highest leverage first):**

1. **Add counterfactual logging** (deterministic-only vs LLM-picked, parallel equity
   curves) — answer "does the LLM help?" with data. *(3.1)*
2. **Invert the conviction model** — gate on empirical/statistical edge; demote the LLM to
   context-veto. *(2.1)*
3. **Make signal promotion evidence-gated** — walk-forward + out-of-sample + multiple-testing
   haircut; holdout **blocks**, not warns. *(3.2)*
4. **Introduce a portfolio layer** — one objective function; size/select against book-level
   correlation, factor, and net exposure. *(1.1)*
5. **Decompose the backtest monolith** behind a shared `Portfolio`/`Broker`/`MarketData`
   abstraction the live path also uses. *(2.2, 2.3)*

**7. Top 5 to NOT change:** the shared canonical signal evaluator (5.1); the fail-closed
execution core (5.2); dormant-by-design self-modification (5.3); structured-tool LLM +
domain validation + audit trail (5.4); deterministic sizing with Kelly as telemetry-only
(5.5).

**8. Recommended next-stage architecture:**

```
Universe ─▶ Data Ports (MarketData/Fundamentals/Options) ── single source of snapshots
                                  │
                                  ▼
              signals.evaluator (CANONICAL — used by backtest, shadow, live)  ◀── keep 5.1
                                  │  raw fired signals + per-signal evidence
                                  ▼
        ┌─ EVIDENCE GATE (statistical edge by signal×regime, walk-forward validated) ─┐
        │                        │                                                    │
        │                        ▼                                                    │
        │           Portfolio Constructor (objective fn: risk-adj return /            │
        │           drawdown budget; correlation + factor + net-exposure aware)       │
        │                        │                                                    │
        │                        ▼                                                    │
        │              LLM = CONTEXT VETO LAYER (news/filings/options → suppress      │
        │              or down-weight; cannot create or upsize)  ◀── reframed 2.1     │
        │                        │                                                    │
        └────────────────────────┼───────────────────────────────────────────────────┘
                                  ▼
            Deterministic Risk Gate + Sizing (fail-closed; Kelly telemetry-only)  ◀── keep 5.2/5.5
                                  │
                    ┌────────────┼─────────────┐
                    ▼            ▼              ▼
                 LIVE        LIVE-SHADOW   DETERMINISTIC-ONLY SHADOW   ◀── new 3.1
                  (real)      (paper)       (counterfactual curve)
                                  │
                                  ▼
        Measured Controller: rolling OOS expectancy + CIs → decay flags →
        evidence-gated promote/demote (human approves; never auto-writes)  ◀── 4.1 + keep 5.3
```

The shape change is: **evidence gates, the LLM vetoes, a portfolio object is the unit of
decision, and three parallel equity curves make the LLM's and each signal's contribution
measurable.** That turns the system from "a very well-built trade-recommendation engine"
into "an evidence-driven research platform that also trades" — which is what the £150
account says it already wants to be.

---

*This is a critique of framing and next-stage design, not of build quality — the quality is
high, which is exactly why it's worth pushing on the strategy.*
