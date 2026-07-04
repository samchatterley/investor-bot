# InvestorBot: Does LLM Contextual Judgement Add Incremental Predictive Value Over a Deterministic Equity Signal Engine?

*A pre-registered study, and the governed measurement system built to answer it.*

> **Status: pre-registration (v1). Data collection not yet begun.** The hypothesis, design, and
> analysis plan are frozen in [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md). Results, Discussion, and
> Conclusion are pending; this document states what will be measured and how, not what was found.
> Pre-committing the analysis before seeing the data is deliberate, because it is the property that
> makes a positive result credible. See [References](#references) on pre-registration. The freeze that
> gates the start of primary collection — the Point of No Return — and what may change afterward are
> specified in [`docs/POINT_OF_NO_RETURN.md`](docs/POINT_OF_NO_RETURN.md).

---

## Abstract

Retail algorithmic trading typically collapses into one of two failure modes: full manual control
(no automation), or opaque model autonomy (no interpretability, override, or audit trail).
InvestorBot occupies the middle ground. A deterministic signal engine generates and risk-filters
candidates, and a large language model (LLM) supplies contextual judgement on top. The project's
central claim, that an LLM's contextual synthesis adds measurable incremental predictive value over a
deterministic signal engine, is treated here as a falsifiable hypothesis rather than an assumption.

We pre-register a three-arm, within-candidate ablation: (1) a frozen deterministic Champion selector;
(2) a structured-only LLM that reasons over identical structured inputs rendered as prose; and (3) a
contextual LLM identical to (2) plus a timestamp-safe context packet of news, filings, and events.
The decomposition isolates the function-approximator effect (Arm 2 minus Arm 1) from the contextual
effect (Arm 3 minus Arm 2), which is the thesis. The primary endpoint is the incremental information
coefficient (IC) of the LLM's explicit, context-driven conviction revision on 5-day forward
risk-normalised return, controlling for the frozen evidence score. Measurement runs on two tracks: a
point-in-time historical backtest (the function-approximator question) and a live-shadow forward test
(the contextual thesis). Both are gated by two Phase-0 checks, an instrument-noise audit and a
statistical-power analysis. A preliminary power projection indicates the live track is underpowered to
resolve a modest effect within a six-month window, so it is pre-scoped as a trend and qualitative
evidence layer rather than the primary statistical test. The deliverable is a governed, auditable
measurement framework and an honest result, including a null, not a profitable trading bot.

---

## 1. Introduction

### 1.1 Background

Automated equity trading systems for non-institutional capital face a credibility gap. Rule-based
systems are interpretable but brittle and easily over-fit; end-to-end ML systems can be powerful but
opaque, hard to govern, and difficult to audit when they act on real money. The recent capability jump
in LLMs reopens an old question in a new form. Discretionary portfolio managers have long been
believed to add value through synthesis and context, weighing many weak and noisy signals against a
live market backdrop. If an LLM can replicate even part of that contextual judgement at scale, that
would be a genuinely novel source of edge. If it cannot, if the LLM merely rides edges already present
in the underlying signals while dressing them in fluent narrative, then the appearance of value is an
illusion that careful measurement should expose.

InvestorBot is the apparatus built to put that question to an honest test. It runs on a deliberately
tiny live account (a roughly £150 `SMALL_ACCOUNT_MODE` profile) used as a realism harness, not a
profit engine. The small scale makes friction and operational reality concrete without pretending the
live P&L could ever reach statistical significance. The LLM is kept central to the system, which is
the thesis, but every part of its authority is designed to be measurable, calibrated, and revocable,
and it is never permitted to place, modify, or cancel orders directly (see
[Appendix A](#appendix-a-system-architecture-and-governance)).

### 1.2 Literature review

**Market efficiency and predictability.** The efficient-market hypothesis (Fama, 1970) sets the null
that public information is already priced. Yet a large literature documents robust cross-sectional
return predictability inconsistent with strong-form efficiency: momentum (Jegadeesh & Titman, 1993),
the size and value factors (Fama & French, 1993), and post-earnings-announcement drift (Ball & Brown,
1968; Bernard & Thomas, 1989). Behavioural accounts (Kahneman & Tversky, 1979; Black, 1986) explain
why such drifts persist. InvestorBot's deterministic signal engine is built from exactly these
documented, commoditised effects, which is precisely why they make a strong baseline rather than a
strawman.

**Machine learning and text in asset pricing.** ML methods improve cross-sectional return prediction
over linear factor models (Gu, Kelly & Xiu, 2020). A parallel strand shows that text carries priced
information: media tone predicts returns (Tetlock, 2007), and domain-specific lexicons and models
matter (Loughran & McDonald, 2011; FinBERT, Araci, 2019). Most recently, LLMs have been applied
directly to return prediction (Lopez-Lira & Tang, 2023; BloombergGPT, Wu et al., 2023).

**Measurement and its pitfalls.** The information coefficient and the fundamental law of active
management (Grinold & Kahn, 2000) provide the natural metric for incremental signal value. But the
field is plagued by multiple testing and backtest overfitting: the cross-section of "discovered"
predictors is statistically implausible without correction (Harvey, Liu & Zhu, 2016), and reported
Sharpe ratios require deflation for selection bias (Bailey & López de Prado, 2014). Pre-registration
is the standard remedy from the wider reproducibility movement (Nosek et al., 2018).

**The gap.** Existing LLM-in-finance work largely asks whether the LLM alone predicts returns. It
rarely asks the question that matters for a system that already has a deterministic engine: does the
LLM add incremental value over that engine, and is that increment attributable to context rather than
to the LLM merely being a better non-linear combiner of the same structured inputs? It also rarely
pre-registers, rarely controls the function-approximator confound, and rarely guards against the
lookahead leakage that makes contextual backtests produce false positives. This study targets that gap
directly.

### 1.3 Research question

> Does an LLM's contextual judgement add measurable incremental predictive value over a deterministic
> equity signal engine, after controlling for (a) the engine's own structured evidence and (b) the
> LLM's non-contextual reasoning ability?

### 1.4 Hypotheses

- **H1 (primary).** For an eligible candidate, the LLM's explicit context-driven conviction revision
  (`context_adjustment`) has positive incremental predictive power on 5-day forward risk-normalised
  return, after controlling for the frozen deterministic evidence score.
- **H2 (decomposition, function approximation).** Arm 2 minus Arm 1: LLM reasoning over identical
  structured inputs adds value over the deterministic score. (Interesting, but not the thesis.)
- **H3 (decomposition, context).** Arm 3 minus Arm 2: adding the context packet, with everything else
  held fixed, adds value. (This is the thesis, isolated.)

Conclusiveness is asymmetric. The design can falsify cleanly; a positive result can only fail to
reject within this universe, signal menu, regime window, baseline version, and horizon, and requires
later replication.

---

## 2. Method

The exact, frozen specifications live in [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md); this section
summarises them.

### 2.1 Design

A three-arm, within-candidate ablation, run at the same decision time on the same eligible candidate
set with the same structured inputs. Arms differ only as below.

| Arm | Description | Extra input | Isolates |
|-----|-------------|-------------|----------|
| 1. Deterministic Champion v1 | frozen non-LLM scorer/selector | (none) | the null |
| 2. Structured-only LLM | structured evidence as prose | (none) | LLM as function-approximator |
| 3. Contextual LLM | Arm 2's exact prompt plus context | `context_packet_v1` | the thesis |

Arm 2 minus Arm 1 gives the function-approximation value; Arm 3 minus Arm 2 gives the contextual
value. Isolation constraint: Arm 3's prompt is Arm 2's exact prompt plus an appended context block;
nothing else differs, and Arm 3 consumes Arm 2's frozen output (so Arm 3 always implies Arm 2 on the
same candidate). Two measurement tracks: a point-in-time historical backtest (Arm 1 versus Arm 2 only,
with no reconstructable context) and a live-shadow forward test (all three arms).

### 2.2 Participants (subjects and agents)

- **Subjects:** the population of eligible candidates, meaning symbols from a ~900-name US-equity
  universe (S&P 500 + S&P 400, large and mid cap, plus ETFs, dynamically extended) that pass the
  canonical signal evaluator and
  the deterministic tradability, liquidity, regime, event, and risk filters on a given decision.
- **Sampling per decision (cost-capped):** all context-present candidates (flagged by a cheap
  deterministic context-presence gate) up to a daily cap; a random control drawn from non-flagged
  eligible candidates (which audits gate false-negatives, and is load-bearing rather than optional);
  plus top-K deterministic candidates for the Arm 2 versus Arm 1 diagnostic.
- **Agents under test:** the deterministic Champion (Arm 1) and the LLM (Anthropic Claude) in
  structured-only (Arm 2) and contextual (Arm 3) configurations.

### 2.3 Materials

- **Data:** OHLCV and intraday bars (Alpaca, yfinance), options-derived features, news, filings,
  sentiment, sector and macro series. Reconstructability is tiered A/B/C in
  [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md) section 6.
- **Deterministic Champion v1 (`evidence_score_v1`):** a frozen, versioned, transparent score (signal
  expectancy, signal-by-regime expectancy, confluence, liquidity, trend quality, minus volatility,
  spread, decay, and known structured event-risk penalties). Component values are computed
  point-in-time; weights are fit and frozen on a train, validation, and holdout split.
- **`context_packet_v1`:** a frozen, mechanically-assembled context bundle (sources, lookback, max
  items, ordering, exclusions), never a hand-curated analyst note.
- **LLM interface:** a structured tool-call schema (typed output), an independent domain validator, and
  a full prompt and response audit trail. The LLM emits an explicit `context_adjustment` (a coarse
  ordinal plus `veto` and `reason_code`), not a noisy difference of two free-form calls.
- **As-of context ledger:** every context item carries source, seen, retrieved, and published
  timestamps and a `timestamp_confidence` label; only items provably seen before the decision are
  admissible.

### 2.4 Procedure

At each decision point the canonical evaluator fires signals, then deterministic filters produce the
eligible set, then the context-presence gate partitions it. The scored set (context-present cap, plus
non-flagged control, plus top-K) is assembled. Arm 1 scores everything; Arm 2 runs on the scored set;
Arm 3 runs on the context-present set and the random control. All outputs and the admissible context
packet are logged. After a fixed horizon, forward R is computed for every scored candidate. Forward R
is path-independent: decision-time entry (not next-day open, which would discard the same-day context
reaction), exit at the split- and dividend-adjusted close H trading days later, normalised by ATR at
decision, costs included. The capital P&L of the tiny live book is never the thesis metric.

### 2.5 Ethics

- **Financial risk and scope.** Live operation uses a deliberately tiny account as a realism harness.
  This is a research system, not investment advice; no third-party capital is solicited or managed.
  Live mode requires an explicit acknowledgement string and extended paper-trading first.
- **Research integrity.** The hypothesis, primary endpoint, and analysis plan are pre-registered and
  frozen before data collection. One primary cell is designated to prevent multiple-comparison fishing;
  test timepoints are fixed to prevent optional stopping (peeking); negative and null results will be
  reported in full. Lookahead leakage, the dominant false-positive risk for a contextual study, is
  controlled by the as-of ledger and a safety buffer.
- **Data and third parties.** Market, news, and filing data are used within provider terms of service;
  no personal data is processed; secrets are environment-only and never logged or committed.
- **Reproducibility.** A credential-free `demo` mode reproduces a full decision cycle on static
  fixtures (see [Appendix D](#appendix-d-reproducibility-and-quick-start)).
- **Conflict of interest.** The author builds and operates the system on their own capital; the framing
  here is FDE-portfolio and research, not a commercial solicitation.

### 2.6 Analysis plan

- **Primary cell (pre-registered, single):** in the context-present bucket, does `context_adjustment`
  predict 5-day forward R after controlling for `evidence_score_v1`, at the pre-registered bar and the
  pre-registered test point? Everything else (other buckets, baselines, horizons, `reason_code`
  breakdowns, veto analysis) is exploratory: reported, but never the headline.
- **Effect decomposition:** overall context value equals P(material context) times the expected effect
  when context is material. A rare but real edge is still real.
- **Phase-0 gates (run first):** (A) an instrument noise audit, where repeated calls on stratified
  snapshots must reproduce the coarse `context_adjustment`; (B) a power analysis, where the minimum
  detectable IC is about z divided by the square root of N_eff, against expected candidate flow (see
  [`scripts/phase0_power_analysis.py`](scripts/phase0_power_analysis.py)). If either fails, the heavy
  live apparatus is not built as v1.
- **Sequential discipline:** monthly outputs are monitoring only; formal tests occur at fixed `N_eff`
  milestones; no headline claim is made before a formal test point.
- **Governance as adaptive protocol:** LLM authority is staged (modes 0 to 5), phased by sample size,
  and revoked only on significance-gated underperformance; rejected-trade triggers require a symmetric
  counterfactual simulation.

---

## 3. Results

> **Pending data collection.** Phase 0 must clear first; the primary cell is tested only at the
> pre-registered `N_eff` milestones. No confirmatory results exist yet.

### 3.1 Phase 0 (Gate B): preliminary power projection (not data)

Using placeholder candidate-flow assumptions (to be replaced by a week of measured logging), the power
tool projects:

| Scenario | N_eff (material) | Min. detectable IC | Verdict |
|----------|------------------|--------------------|---------|
| Baseline (small-account flow, about 6 months) | about 142 | about 0.165 | underpowered |
| Optimistic (3x eligibility, full year) | about 1,089 | about 0.059 | still underpowered for IC = 0.05 |

A plausible single-feature incremental IC (0.03 to 0.05) needs an N_eff of roughly 1,500 or more.
Implication (pre-registered): the live contextual track is scoped as a trend and qualitative evidence
layer; the historical Arm 1 versus Arm 2 ablation and the governed framework itself are the primary v1
outputs. This is a projection from assumptions, not a study result.

---

## 4. Discussion

> **Pending results.** Recorded here in advance are the threats to validity the design controls for, so
> interpretation is pre-committed rather than rationalised after the fact.

- **Function-approximator confound:** addressed by the Arm 2 ablation (H3 is Arm 3 minus Arm 2).
- **Omitted-variable confound:** structured event existence lives in `evidence_score_v1`; only its
  interpretation is credited to Arm 3.
- **Anchoring bias:** eliciting an explicit revision from Arm 2's frozen output reduces variance but
  biases toward zero, so a positive result is conservative and a null is ambiguous.
- **Lookahead leakage:** the as-of ledger and safety buffer prevent the contextual arm's chief
  false-positive vector.
- **Baseline strength:** any win is reported as a win versus `evidence_score_v1`; a stronger fitted
  baseline (v2) is pre-committed for a later, separate period.
- **External validity:** a positive result generalises only within this slice and demands replication
  across horizon, regime, signal family, and baseline version.

---

## 5. Conclusion (interim)

No thesis verdict is claimed. The contribution to date is methodological: a pre-registered design and a
governed, auditable, lookahead-controlled measurement framework that can falsify the
AI-contextual-value claim, plus tooling that already quantifies the study's own statistical limits.
Under the project's FDE-first framing, that framework, and an honest result including a null, is the
deliverable, independent of which way the number eventually lands.

---

## References

1. Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.* arXiv:1908.10063.
2. Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management*, 40(5).
3. Ball, R., & Brown, P. (1968). An Empirical Evaluation of Accounting Income Numbers. *Journal of Accounting Research*, 6(2).
4. Bernard, V. L., & Thomas, J. K. (1989). Post-Earnings-Announcement Drift: Delayed Price Response or Risk Premium? *Journal of Accounting Research*, 27.
5. Black, F. (1986). Noise. *The Journal of Finance*, 41(3).
6. Fama, E. F. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work. *The Journal of Finance*, 25(2).
7. Fama, E. F., & French, K. R. (1993). Common Risk Factors in the Returns on Stocks and Bonds. *Journal of Financial Economics*, 33(1).
8. Grinold, R. C., & Kahn, R. N. (2000). *Active Portfolio Management* (2nd ed.). McGraw-Hill.
9. Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *The Review of Financial Studies*, 33(5).
10. Harvey, C. R., Liu, Y., & Zhu, H. (2016). ...and the Cross-Section of Expected Returns. *The Review of Financial Studies*, 29(1).
11. Jegadeesh, N., & Titman, S. (1993). Returns to Buying Winners and Selling Losers. *The Journal of Finance*, 48(1).
12. Kahneman, D., & Tversky, A. (1979). Prospect Theory: An Analysis of Decision under Risk. *Econometrica*, 47(2).
13. Lopez-Lira, A., & Tang, Y. (2023). *Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models.* arXiv:2304.07619.
14. Loughran, T., & McDonald, B. (2011). When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks. *The Journal of Finance*, 66(1).
15. Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018). The Preregistration Revolution. *PNAS*, 115(11).
16. Tetlock, P. C. (2007). Giving Content to Investor Sentiment: The Role of Media in the Stock Market. *The Journal of Finance*, 62(3).
17. Wu, S., et al. (2023). *BloombergGPT: A Large Language Model for Finance.* arXiv:2303.17564.

---

## Appendices

### Appendix A: System architecture and governance

InvestorBot is an AI-governed execution-control system for US equities. Claude performs analysis and
issues structured recommendations; a deterministic validator, risk gate, and human-override layer
decide whether to act. Claude can never place, modify, or cancel orders; read or write config; access
balances or positions; or modify its own parameters. Every output is schema-validated and risk-gated
first.

- **Decision pipeline:** market context, then Claude (a single structured tool call that ranks both
  long *and* short candidates — shorts are rule-gated in the bear regimes and routed through the AI for
  the same veto/ranking as longs), then two-layer
  validation (API schema, plus a domain whitelist, confidence, conflict, and injection scan), then the
  risk gate (risk-budget sizing at 0.6% of equity risked per trade, capped at 15% of portfolio per
  position; position and sector limits, fat-finger and daily-notional guards, bear
  filter, VIX-tiered stops, earnings guard, same-day churn guard, circuit breaker, daily-loss limit; shorts
  add sector-momentum, correlation, borrow-cost, and squeeze gates plus standalone-vs-hedge caps), then execution
  (fractional market orders plus a trailing stop), then an append-only SQLite audit.
- **Module map:** `data/` (market, news, options, sentiment, macro); `signals/` (the canonical
  evaluator); `analysis/` (ai_analyst, weekly_review, performance); `risk/` (sizing, calendars,
  checks); `execution/` (trader, scanner, universe); `backtest/`; `utils/` (audit, ledger,
  validators); `notifications/`. Orchestrated by `main.py`, scheduled by `scripts/run_scheduler.py`.
- **Constrained parameter engine:** the weekly review proposes bounded parameter changes; they are
  validated, logged, and emailed but never auto-applied (the operator edits `logs/runtime_config.json`).
- **Human override:** `python cli.py halt` and `resume`; the halt file is checked before every cycle.

Full detail: [`docs/adr/`](docs/adr/) (design decisions), [`docs/signals.md`](docs/signals.md) (signal
catalogue), [`docs/strategic_review.md`](docs/strategic_review.md) (architecture critique),
[`docs/audit_v1.100.md`](docs/audit_v1.100.md) (line-by-line audit), and [`LIVE_RUNBOOK.md`](LIVE_RUNBOOK.md).

### Appendix B: Prior evaluation evidence (deterministic backtest)

The rule-based backtester is a proxy for signal quality; it does not call Claude. Combined long/short,
2015-01-01 to 2026-06-12, $25k starting capital: total return +31.0%, 5,256 trades, 51% win rate,
average +0.06% per trade, max drawdown -37.8%, Sharpe 0.22. Caveats: it is a rule-based proxy, not live
Claude; transaction costs are modelled (slippage, a liquidity-scaled spread, and market impact); there
is no lookahead (T-1 indicators, T-open fills). Survivorship bias (the universe is fixed to current
S&P 500 + S&P 400 constituents) and fundamental look-ahead in the distress shorts mean pre-2020 and short-book
results are upward-biased. This is exactly the kind of in-sample evidence the pre-registered study is
designed to supersede.

### Appendix C: Engineering rigour

- **Tests:** 4,964 tests, 100% line and branch coverage, enforced on CI; the mypy gate is clean across
  the typed modules.
- **LLM eval fixtures** ([`evals/`](evals/)): prompt-injection headlines, hallucinated tickers,
  bear-market no-buy, conflicting signals, earnings-risk, malformed tool calls.
- **Production incident log** ([`docs/incidents.md`](docs/incidents.md)): six day-one paper-trading
  failures, all diagnosed from structured logs alone.
- **FDE relevance:** an ambiguous problem turned into a governed product; untrusted-AI-output handling
  (schema and domain validation plus an audit trail); several fault-tolerant third-party integrations;
  an operator CLI and dashboard; a pre-registered, measurable, falsifiable experimental design.

### Appendix D: Reproducibility and quick start

```bash
git clone https://github.com/samchatterley/investor-bot
cd investor-bot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python cli.py demo            # full simulated cycle on static fixtures, no credentials needed
```

Operate with `python cli.py status | positions | trades | decisions | run --dry-run | halt | resume |
backtest | dashboard`. Live deployment, `.env` keys, Docker, scheduler, monitoring, and the runbook are
documented in the [Appendix A](#appendix-a-system-architecture-and-governance) sources and in
[`LIVE_RUNBOOK.md`](LIVE_RUNBOOK.md). Run the power gate with `python scripts/phase0_power_analysis.py`.

### Appendix E: Version history

See [`CHANGELOG.md`](CHANGELOG.md) (v1.0 to v1.103). Most recent: v1.103 retired two dead FRED series
(AAII-via-FRED, now sourced from aaii.com; and ISM PMI/NAPM, removed outright). v1.102 widened the
tradeable universe to S&P 500 + 400 (large + mid cap, ~900 names) after an edge check validated the
engine down to mid cap, with small caps excluded; v1.101 was a data-feed integrity sweep (a new
feed-health gate plus repairs to four silently-degraded feeds) and experiment material-context
coverage (all ten categories wired). v1.100 was the 100th release, a full line-by-line audit plus mypy cleanup
([`docs/audit_v1.100.md`](docs/audit_v1.100.md)). The research-program reframing
(this document, [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md), and
[`docs/strategic_review.md`](docs/strategic_review.md)) supersedes the prior feature-manual README.
