# EXPERIMENT.md — Pre-registration: Does AI contextual judgement add incremental value?

> **Status:** Pre-registration (v1.3, pre-data amendments). Freeze before collecting any primary
> data. Any change after data collection begins requires a new version and a new evaluation period.
> Companion docs: `docs/strategic_review.md` (why), `docs/audit_v1.100.md` (code-level audit).
>
> **v1.3 amendment (pre-data): LLM lookahead control (section 15.8).** The model has a training
> knowledge cutoff (~Jan 2026), so on historical dates before it the LLM may already know the
> realised outcome (training-data lookahead, affecting Arm 2 as well as Arm 3). Therefore the
> contextual thesis is live-forward only (a hard requirement), and the historical Arm1-vs-Arm2
> ablation feeds Arm 2 only blinded, de-identified features. Holdout boundary stays 2024-01-01 (a
> longer, multi-regime out-of-sample window for the fitted v2).
>
> **v1.2 amendment (pre-data): the standable baseline is the fitted model.** To make a positive
> result one we can stand on, the headline benchmark is the fitted `evidence_score_v2` (section 15.7),
> not the transparent hand-weighted `evidence_score_v1`. v1 is the floor and interpretability
> diagnostic only. All baseline expectancy components must be computed point-in-time (as-of each
> decision); zero-defaulting an absent expectancy is forbidden (it produces a strawman baseline that
> also leaks). The AI claim requires beating v2.
>
> **v1.1 amendments (no data collected yet, so still a legitimate pre-registration).** Adopted from
> an external audit of the design: (1) the primary endpoint is recast as a veto / down-weight test on
> material-context candidates (sections 1 and 3); (2) more research snapshots than trades, with
> clustered standard errors (section 15.2); (3) negative and positive controls (section 15.3);
> (4) pre-registered economic-usefulness thresholds (section 15.5); (5) a narrow,
> high-timestamp-integrity historical context track (section 15.6). Deferred items are in section 16.

---

## 0. One-paragraph statement

InvestorBot's central claim is that **an AI's contextual synthesis adds measurable incremental
predictive value over a deterministic signal engine.** We treat this as a *falsifiable
hypothesis*, not an assumption. The AI stays central to the system, but every part of its
authority must be **measurable, calibrated, and revocable**. The £150 live account is a realism
harness, not the statistical proof engine. This is an **FDE-first** project: the deliverable is a
governed measurement framework + an honest result (including a null or "not yet resolvable"),
**not** a profitable bot.

---

## 1. The hypothesis (and what would falsify it)

**H1 (primary, v1.1): the veto / down-weight test.** Among deterministic-eligible candidates that
carry a pre-registered material-context category (section 15.1), the candidates the contextual arm
(Arm 3) vetoes or marks with a negative `context_adjustment` underperform the non-vetoed candidates
on forward risk-normalised return, after controlling for `evidence_score_v1` and the structured event
flags. This is the headline because it is the highest signal-to-noise, most operationalisable, and
most directly actionable form of the thesis: it yields a governance decision (enable the veto, disable
it, or keep the LLM advisory-only).

**H2 (secondary): the incremental-IC test (the former primary).** Across material-context candidates,
`context_adjustment` has positive incremental predictive power on 5-day forward risk-normalised return
after controlling for `evidence_score_v1`.

**H3 (decomposition): context vs reasoning.** Arm 3 minus Arm 2 isolates the value of unstructured
context (the thesis); Arm 2 minus Arm 1 isolates LLM reasoning over identical structured inputs (not
the thesis).

- **Falsified** if vetoed candidates do not underperform non-vetoed ones at the pre-registered test
  point (a clean, informative negative result that disables veto authority).
- **Supported (not proven)** if they do, claimable only within this universe, signal menu, regime
  window, baseline version, and horizon. Positive results require later replication.

Conclusiveness is asymmetric: we can falsify cleanly; we can only fail to reject on a narrow slice.
The artifact is the governed measurement, whichever way the number lands.

---

## 2. The three arms

All arms run on the **same eligible candidate set** at the **same decision time**, with the **same
structured inputs** (fired signals, structured features, regime labels, risk/liquidity filters,
and the frozen `evidence_score`). They differ only as below.

| Arm | What it is | Inputs | Isolates |
|-----|------------|--------|----------|
| **Arm 1 — Deterministic Champion v1** | Frozen non-LLM selector/scorer | structured evidence only | the null hypothesis |
| **Arm 2 — Structured-only AI** | LLM, structured evidence as prose | same evidence, no context | LLM-as-function-approximator |
| **Arm 3 — Contextual AI** | Arm 2's *exact* prompt + appended context | structured evidence + `context_packet_v1` | **the thesis** |

**Decomposition:**
- `Arm 2 − Arm 1` = value of LLM reasoning over identical structured inputs *(not the thesis)*.
- `Arm 3 − Arm 2` = value of unstructured context, everything else held fixed *(the thesis)*.

**Isolation constraints (mandatory):**
- Arm 3's prompt **= Arm 2's exact prompt + an appended context block**. Nothing else differs.
- The structured-evidence prose is a **fixed, mechanical, versioned template** — never hand-curated.
- **Arm 3 ⟹ Arm 2** on the same candidate (Arm 3 consumes Arm 2's frozen output — it is
  uncomputable without it). Arm 2 may run standalone (for the `Arm 2 − Arm 1` diagnostic).

---

## 3. The single pre-registered primary cell

To avoid multiple-comparison fishing across buckets, baselines, horizons, and reason-codes, exactly
one cell is the headline test. Everything else is exploratory or robustness: reported, never the
headline.

> **PRIMARY (v1.1):** Among deterministic-eligible candidates carrying a pre-registered
> material-context category (section 15.1), do the candidates that Arm 3 vetoes or marks
> `context_adjustment = negative` underperform the rest on 5-trading-day forward R, after controlling
> for `evidence_score_v1` and the structured event flags, at the pre-registered bar (section 9) and
> test point (section 10)?

The material-context categories are pooled into this one cell; per-category effects (by `reason_code`)
are exploratory. The former primary, the incremental IC of `context_adjustment` (H2), is the lead
secondary endpoint.

---

## 4. Primary metric — forward R (path-independent)

The primary test is **predictive**, so forward R is a clean H-day forward return per risk unit, not
a realised stop/target outcome (that is a *secondary realism* metric only).

```
H              = 5 trading days (primary; 1/3/10/strategy-exit are labelled robustness only)
entry_price    = price at decision_time  (decision-time fill / short post-decision VWAP)
                 ── NOT next-day open: decisions are made intraday (~10:00 ET); next-day open
                    discards the same-day context reaction and biases the test toward null.
                    next-open R is retained as a SECONDARY realism metric.
exit_price     = split/dividend-adjusted close, H trading days after entry
risk_unit      = ATR14 at decision_time
costs          = estimated spread + slippage + fees

long:   forward_R = (exit_price - entry_price - costs) / risk_unit
short:  forward_R = (entry_price - exit_price - costs) / risk_unit
```

Forward R is computed for **every** eligible candidate (traded or not) — this is the measurement
overlay, not the traded book. Capital P&L of the £150 book is **never** the thesis metric.

**Secondary metrics (exploratory):** matched-decision ΔR vs Champion's pick; parallel shadow equity
curves (Arm 1 / Arm 2 / Arm 3); win-rate uplift; avg winner/loser; MAE; Sharpe/Sortino; veto
performance; confidence calibration (Brier, reliability curve); per-signal and per-regime uplift;
per-`reason_code` breakdown.

---

## 5. Deterministic Champion v1 — the baseline

A frozen, versioned selector that receives the same fired signals, structured features, regime
labels, and risk/liquidity filters as the AI, but **no news/filings/narrative/LLM**. Two roles:
**scorer** (for the IC test) and **selector** (highest score, for realism arms). Portfolio/path-
dependent filters (position limits, concentration) attach to the *selector* only.

```
evidence_score_v1 =
      signal_edge_score          # historical (point-in-time) expectancy of this signal
    + signal_regime_score        # point-in-time expectancy of signal × current regime
    + confluence_score           # bonus for multiple independent signals firing
    + liquidity_score            # tighter/more-liquid names score higher
    + trend_quality_score        # structured price/volume confirmation
    - volatility_penalty         # unstable names with poor stop geometry
    - spread_penalty             # expensive execution
    - decay_penalty              # point-in-time OOS degradation of the signal
    - event_risk_penalty         # KNOWN structured events (earnings soon, etc.) — see §8
    # concentration_penalty: DROPPED from v1 (portfolio/path-dependent → later portfolio layer).
```

**Provenance rules (anti-lookahead, anti-overfit):**
- All expectancy/decay component *values* are computed **point-in-time** (as-of the decision, prior
  data only) — not full-sample.
- Component **weights** are fit and **frozen** on a train → validation → **holdout** split, before
  evaluation. Never tuned after seeing Arm 3 results on the same period.
- Versioned object `evidence_score_v1`; the AI is always measured against the version live at
  decision time. New version → new evaluation period.

**Baseline ladder** (only v1 is the headline; others are diagnostic): random eligible → signal-
priority → **evidence_score_v1 (primary)** → evidence_score_v2 (later fitted model, own period) →
future champion.

---

## 6. `context_packet_v1` — the treatment (frozen)

The context packet IS the treatment; if it changes freely the experiment is invalid. Mechanically
assembled, versioned, never a hand-written analyst note.

```yaml
context_packet_v1:
  lookback_window: 48h
  max_items_per_symbol: 8
  allowed_context:
    - timestamp-safe news headline/snippet
    - filing metadata/excerpt
    - earnings/event calendar item
    - analyst action headline (if timestamp-safe)
  ordering: [filings, earnings/events, symbol-specific news, sector/macro]
  exclusions:
    - items seen after decision_time
    - unclear/ambiguous timestamps
    - provider backfills
    - generated summaries not traceable to a raw source
  empty_behaviour: append an explicit "no admissible context" block (do not omit silently)
```

---

## 7. Arm 2 / Arm 3 output contracts

**Arm 2 (structured-only)** emits: `structured_conviction` (coarse — see noise audit),
`structured_rank`, `structured_reasoning`.

**Arm 3 (contextual)** receives the same structured packet **+ Arm 2's frozen output +
`context_packet_v1`**, and emits the explicit revision (the primary instrument — a direct answer,
not a noisy difference of two independent calls):

```
context_adjustment   : starts COARSE — {negative, neutral, positive}  (+ veto)
                       refine to {-2,-1,0,+1,+2} ONLY if the noise audit (§9) supports it
context_materiality  : {0,1,2,3}
veto                 : bool        (handled as its own outcome, not folded into the sum)
reason_code          : {news, filing, earnings, macro, sector, options, other}
final_conviction     : structured_conviction (+) context_adjustment   # mechanical
```

`Arm 3 − Arm 2` is still logged, but **`context_adjustment` is the primary variable.** Note the
deliberate trade: anchoring Arm 3 on Arm 2's output reduces variance at the cost of a *conservative
(toward-zero) bias* — a positive result is therefore trustworthy; a null is ambiguous.

---

## 8. Sampling, the context-presence gate, and validity controls

**Buckets (pre-registered):**

| Bucket | Meaning | Role |
|--------|---------|------|
| All eligible | every candidate passing filters (or a capped sample) | base-rate |
| **Context-present** | timestamp-safe context exists (gate fired) | **primary thesis sample** |
| Context-decisive | AI says context materially changed the view | high-signal secondary (selection-biased) |

**Overall context value = P(material context) × E[effect \| material].** A rare-but-real edge is
still real.

**Context-presence gate** (cheap deterministic pre-filter, run before paying for Arm 3): recent
filing, earnings in ±N days, abnormal gap, abnormal volume, major news item, analyst action,
short-interest/borrow event, sector shock, macro-sensitive day. Note this selects candidates the
structured layer already flags → the primary claim is "context value **at structurally-flagged
events**." A `gate-fired-but-empty-after-timestamp-filtering` candidate is tagged separately (not
the same as non-flagged).

**Random control = random sample of eligible-but-NON-flagged candidates** (run both arms). Its job:
detect gate false-negatives (real context with no structural precursor) and supply
`E[effect | not flagged]`. It is **load-bearing, not optional.** Tracked in a `context_gate_audit`
report (flagged_count, random_control_count, material_context_in_random_controls, est. false-neg rate).

**Omitted-variable control:** structured event *existence* belongs in `evidence_score_v1`
(`event_risk_penalty` etc.); only the *qualitative interpretation* of the event belongs to Arm 3.
Audit `evidence_score_v1` for obvious structured event variables before the primary test, so Arm 3
gets credit for interpretation, not for discovering a missing deterministic flag.

**Daily scored set (cost-capped):**
```
scored_union = context_present(cap) ∪ random_control(non-flagged) ∪ top_K_deterministic
for c in scored_union:
    run Arm 2
    if c ∈ context_present ∪ random_control:
        run Arm 3            # implies Arm 2 already ran
# top_K-only candidates: Arm 2 only (feeds the Arm 2 − Arm 1 diagnostic)
```

---

## 9. Phase 0 — the two go/no-go gates (do these FIRST, in week 1)

Both gate whether the heavy live apparatus is worth building.

**Gate A — Noise audit (is the instrument stable?).** 25–50 candidate/context snapshots,
**stratified across expected materiality** (clearly-material / neutral / borderline — borderline is
where noise bites). Fixed model, temperature, prompt, `context_packet_v1` draft. **5 repeated runs**
per candidate. Measure sign stability, veto stability, reason-code stability, conviction variance,
arbitrary-flip rate.
- **Fail** ⇒ the coarse instrument can't reproduce its own judgement → drop granularity further
  (veto/no-veto, or pairwise preference) or pivot to qualitative case studies. Do not run a
  conviction-IC experiment on a noisy instrument.

**Gate B — Power analysis (can the sample detect a plausible effect?).** See
`scripts/phase0_power_analysis.py`. Estimate context-present obs/day → raw N → apply
clustering/neutral-context discount → `N_eff` → min detectable IC ≈ `z / √N_eff`.
- **Decision rule:** if the target window can only detect an implausibly large IC (e.g. min
  detectable IC ≫ a plausible single-feature edge of ~0.03–0.05), **downgrade the live track** from
  "primary statistical thesis test" to "trend + qualitative evidence layer." That is honest scoping,
  not failure.

If either gate fails: **do not** build the year-long live infrastructure as v1. Keep a *thin* live
context ledger (useful for the FDE story + future work) and ship the pivoted v1 (§11).

---

## 10. Testing schedule (no peeking)

Pre-register *when*, not just *what*. The live track accumulates over time, so unscheduled looks
inflate false positives.

- **Monthly reports:** descriptive **monitoring only** — explicitly **not** hypothesis tests.
- **Interim test:** at `N_eff ≥ 200` (or date X) — non-binding.
- **Primary v1 test:** at `N_eff ≥ 400` (or date Y).
- **No headline claim before a formal test point.** Fixed milestones suffice for v1 (no
  alpha-spending needed at this look-count).

---

## 11. Scope — MVP v1 (guard against building-instead-of-shipping)

The dominant practical risk is six months of elegant infrastructure and no result. v1 is small.

**Build in v1:** this `EXPERIMENT.md`; `evidence_score_v1`; `context_packet_v1`; Phase 0 noise audit;
Phase 0 power analysis; historical Arm 1-vs-Arm 2 ablation; a thin timestamp-safe live context
ledger; first interim report.

**Do NOT build in v1:** full ingestion service; portfolio layer; broker/data ports; monolith
refactor; automatic authority modes; new signal expansion; multi-baseline championship;
multi-horizon statistical framework.

**Sequence:** items 1–5 (codify + Phase 0) are a **hard gate** on items 6–8.

**v1 deliverable claim:** "I designed and ran a governed measurement framework, determined whether
the live contextual thesis is statistically feasible at this scale, tested LLM structured-only
reasoning historically, and began a timestamp-safe live context ledger."

---

## 12. Governance — authority is measurable, calibrated, revocable

- LLM confidence is **telemetry until calibrated**; interim entry is gated by deterministic evidence.
  Arm 3 selects/ranks/vetoes within the eligible set; no direct sizing authority; no live signal
  invention.
- **Authority modes 0–5** (shadow → commentary → veto → select → select+conviction → portfolio).
  Decompose powers (rank/select/veto/sell) and measure each separately; prioritise **veto + sells**
  (highest signal-to-noise).
- **Phased by sample size:** <50 descriptive · <100 warn · <200 provisional changes · 200+
  statistical. Revocation is **significance-gated**, not twitchy; rejected-trade triggers require the
  rejected trade to be simulated with identical entry/stop/exit/cost/sizing assumptions.

---

## 13. Lookahead safety — the as-of context ledger

Every context item stores: `symbol, decision_id, decision_time, source, source_id, provider_published_at,
provider_seen_at, retrieved_at, included_in_prompt_at, raw_text_hash, raw_payload_snapshot,
timestamp_confidence{clean|uncertain|rejected}`.

**Admissible** only if `provider_seen_at ≤ decision_time − safety_buffer` (1–5 min). **Never query
after the decision and pretend the result pre-dated it** — ingest continuously into the ledger and,
at decision time, read the *ledger*, not the provider. Missing/ambiguous/backfilled timestamps ⇒
excluded from the primary metric. The continuous ingestion service is a **monitored, first-class
dependency** (downtime = lost context-present samples).

---

## 14. Claim language (pre-committed)

- Negative: *"We found no evidence that context adjustments predicted residual returns in this setup."*
- Positive: *"In this universe, signal menu, regime window, baseline version, and horizon, context
  adjustments showed positive incremental predictive value."* → requires later replication across
  horizon / regime / signal family / baseline version / time window.
- Always: *"This is a governed measurement system, not a universal proof of market alpha."*

---

## 15. v1.1 amendments (pre-data, adopted from the experiment audit)

Adopted before any data collection to raise statistical conclusiveness and trading usefulness. These
amend, and where noted supersede, the sections above.

**15.1 Material-context categories (supersedes the generic context-present primary).** The primary and
secondary tests run only on candidates carrying at least one pre-registered material-context category,
which raises signal density by removing the dilution of irrelevant context. Frozen category list:
earnings surprise or drift; guidance raise or cut; insider cluster buying; secondary offering or
dilution; FDA, legal, or regulatory event; index inclusion or deletion; major analyst initiation,
upgrade, or downgrade; M&A rumour or confirmation; accounting or auditor concern; short-squeeze setup.
A candidate is material-context when a deterministic detector assigns it one of these categories. The
categories are pooled for the primary; per-category is exploratory.

*Feed coverage (as of 2026-06-15).* All ten categories now have a live point-in-time feed:
earnings, guidance, insider cluster, analyst action, and short-squeeze through the signal book;
secondary offering, M&A, accounting concern, and regulatory event through EDGAR 8-K item codes
(item 2.01/1.01 for M&A, 4.02/4.01 for accounting, 3.01/8.01 for regulatory; the two broad items are
keyword-confirmed); and index inclusion/deletion through news-headline detection
(`data/index_membership.py`), since the index providers expose no clean point-in-time API but the
events are reliably newsworthy. Feed health is checked by `scripts/feed_health_check.py` before each
collection window; note that several of these feeds are EDGAR/live-only or news-only and so are absent
from the historical backtest sample, which is therefore dominated by the earnings category. The
insider-cluster feed was repaired on 2026-06-15 (Form 4 XSL-path bug), so any pre-fix historical tally
undercounts that category.

**15.2 Research snapshots, not more trades (raises N without changing the strategy).** The live-shadow
track records multiple timestamped research observations per day (for example open plus 30 minutes,
midday, close minus 30 minutes, post-close), each scored by all arms but only a subset ever tradable.
The same ticker recurring across snapshots and days creates dependence, so all inference uses
clustered standard errors by ticker and by date, or a block bootstrap. Target order of magnitude:
thousands of shadow-scored observations over a year, not hundreds.

**15.3 Negative and positive controls (credibility and leakage guard).** Negative controls, which must
show no effect: future-shuffled context, randomly reassigned context packets, stale-only context,
permuted ticker labels, random `reason_code`. If the contextual arm works on any of these, treat it as
leakage or overfitting and halt. Positive controls, which must be detectable or the harness is not
sensitive enough to test the thesis: a momentum signal, an earnings-surprise or PEAD proxy, an
illiquidity or high-spread execution penalty, and elevated MAE under high realised volatility.

**15.4 Secondary endpoints (logged, not headline).** 1-day and 3-day forward ATR return; max adverse
excursion; max favourable excursion; stop-hit probability; gap-against probability; realised
volatility; hit rate conditional on the adjustment; top-decile-minus-bottom-decile spread. The veto's
value is expected to show most in the risk endpoints (MAE, stop-hit, gap-against), not in mean return.

**15.5 Economic-usefulness thresholds (define before significance).** A result must clear an economic
floor, tied to a trading decision, to count as useful: incremental IC above 0.03; veto avoided loss
above 0.30 ATR on average; stop-hit reduction above 10 percent relative; Brier improvement above 5
percent versus baseline; shadow-portfolio Sharpe uplift above 0.2; drawdown reduction above 10 to 15
percent. Statistical significance without clearing the floor is reported as "real but not economically
useful."

**15.6 Narrow historical context track (`context_packet_historical_v0`).** To add power without unsafe
reconstruction, build a thin historical context set from high-timestamp-integrity sources only: SEC
filings, earnings dates and results, and scheduled macro events. Exclude social sentiment, revised
fundamentals, and anything whose as-of time is ambiguous. A narrow but clean historical Arm-3-like
dataset of thousands of observations is worth more than a rich live one with no power.

**15.7 Strong fitted baseline is the headline benchmark (v1.2).** `evidence_score_v2` is a regularised
model on the same point-in-time structured features (logistic, ridge, or gradient-boosted trees), fit
on a training window and frozen on a holdout. It is the benchmark the AI must beat for a standable
claim. `evidence_score_v1` (transparent, hand-weighted, frozen) is the floor and interpretability
diagnostic, not the headline. If Arm 2 or Arm 3 beats v1 but not v2, the LLM is a mediocre non-linear
combiner, not an intelligence layer, and the result is reported as such.

Rigour requirements that follow:
- **Point-in-time only.** Expectancy components (signal_edge, signal_regime, decay) are computed
  as-of each decision from prior data; full-sample expectancy is forbidden (it both inflates the
  baseline and leaks). A candidate without a real point-in-time expectancy is not scoreable for the
  experiment; the scorer flags this rather than zero-defaulting (no silent strawman).
- **Shared point-in-time dataset.** v1, v2, and the historical Arm1-vs-Arm2 ablation all run on one
  point-in-time feature-and-outcome dataset built from the backtest engine, so the baseline, the
  benchmark, and the ablation are mutually consistent and lookahead-clean.
- **Separate periods.** v2 is fit on training, selected on validation, and judged on holdout; the AI
  comparison runs on the holdout / live-forward only.

**15.8 LLM lookahead control: blinding and the model knowledge cutoff (v1.3).** The LLM has a training
knowledge cutoff (about January 2026), so for any historical date before it the model may already know
the realised outcome. This is training-data lookahead and it affects Arm 2 as well as Arm 3.
Consequences:
- **Contextual thesis is live-forward only.** A hard rigour requirement, not just a reconstructability
  preference: there is no clean historical contextual arm.
- **Historical Arm1-vs-Arm2 ablation blinds Arm 2.** Arm 2 receives only de-identified, normalised
  features (no ticker, no date, no absolute price; RSI, MACD-diff, BB%, vol-ratio, relative strength,
  etc.), so the model cannot recall which security and period it is and ride the known outcome. Arm 1
  (deterministic) is unaffected.
- **Blinding leakage probe (control).** Periodically ask the blinded model to name the ticker and
  approximate date from the features alone. If it can, blinding has failed and that batch is excluded.

**15.9 Frozen prompt; the self-learning loop is a separate ablation (v1.3).** The bot's prompt has two
outcome-derived blocks — last week's review *lessons* and a regime-aware *performance-feedback* block
(`config.ADAPTIVE_PROMPT_ENABLED`, gated in `analysis/ai_analyst.build_prompt`). Left on, these mutate
the contextual arm weekly from realised outcomes, which (a) makes Arm 3 non-stationary so per-decision
IC/veto cannot be pooled across weeks, and (b) fits the prompt to the very sample being measured — the
overfitting trap pre-registration exists to prevent. At a ~£150 account a weekly "lesson" is drawn
from a handful of trades and is almost entirely noise, so the loop is unlikely to be learning signal.
Therefore:
- **Core context window: the prompt is frozen** (`ADAPTIVE_PROMPT_ENABLED = False`) so Arm 3 is
  stationary and the Arm-3-minus-Arm-2 contextual test is clean. The Arm-3 prompt (including the
  documented material-context flags) is a versioned artifact; a better prompt is a pre-registered new
  version, never a silent mid-stream edit.
- **The loop's own value is a separate, pre-registered ablation:** frozen Arm 3 vs Arm 3 + lessons.
  Each decision logs which mode was in effect (`adaptive_prompt` in the
  `experiment/collection.py` observation), so the two questions — does *context* help, and does *weekly
  learning* help — are answered one at a time instead of confounding each other.

---

## 16. Deferred to v2 or conditional (explicitly not built in v1)

The audit's own top finding was overbuilding risk. These are deferred to protect against it.

- **Rich quantitative LLM output** (probability forecasts, base/bull/bear cases, sizing multiplier):
  conditional on Gate A showing the coarse instrument is stable. If it passes, add one continuous field
  (for example `prob_positive_5d`) and calibrate it; do not add a full forecast schema up front. A rich
  output before Gate A passes would reintroduce the exact instrument noise the gate exists to catch.
- **Shadow-portfolio factory:** at most three parallel paper books later (Champion, Contextual,
  Contextual-veto-only) as economic interpretation, not the main statistical test. Not seven.
- **Universe expansion** (Russell 1000 or 2000) for event density: research-only, later, after the
  snapshots (15.2) and the historical track (15.6) are exhausted as cheaper sources of N.
- **Ranking-metric frame:** adopt ranking as closer to how the bot acts, but keep one primary ranking
  metric, not a suite of five.
- **Bayesian governance tracking:** a lightweight posterior on the veto effect for promote / demote
  decisions, later; not a parallel inferential framework.
- **Two-stage architecture** (LLM emits structured event and materiality labels, a deterministic
  policy maps labels to action): the right long-run shape, and more robust than trusting raw LLM
  conviction. Pilot it through load-bearing `reason_code` to action mapping, then graduate after the
  veto result is in.
