# Lazy Prices signal + rich unstructured-context scoping

Status: **signal primitive built + live-validated** (`data/filing_similarity.py`, 2026-06-29). Not yet
wired into trading. This doc covers (1) how to backtest Lazy Prices honestly and (2) the forms of
"rich unstructured context" we could feed the LLM and how.

## 1. Lazy Prices — what's built and what's next

`get_filing_change(symbol)` fetches a company's two most recent 10-Ks from EDGAR and returns the
cosine similarity of their narrative text. Low similarity = big year-over-year language change =
bearish (Cohen-Malloy-Nguyen 2020). Live check on large-caps: AAPL 0.995, MSFT 0.974, NVDA 0.989,
INTC 0.965, WBD 0.986 — all high (most firms copy-paste), exactly as expected; the edge is in the
low-similarity tail.

**Two things to settle before it can trade:**

1. **Threshold → cross-sectional quintiles, not an absolute cutoff.** Live scores cluster at
   0.96–0.995, so the provisional `0.80` flag rarely fires. The paper ranks the *cross-section* each
   period and shorts the most-changed quintile. The backtest should compute `change_score` for the
   whole universe per filing season and rank, not threshold.

2. **Horizon mismatch — this is the big one.** Lazy Prices is a *slow* signal: annual filings, drift
   measured over **1–6 months**. Our bot holds **1–5 days**. So it does NOT fit as a swing entry, and
   running it through the 5-day matrix/event-study would (correctly) show nothing. Realistic fits:
   - **Risk flag / de-risk overlay** (best near-term): when a name we hold or are about to buy has a
     high `change_score`, size down or pass. Cheap, no horizon problem, fail-safe.
   - **Slow tilt** (longer hold): a small sleeve held weeks, separate from the swing book.

### Backtest plan (the honest version)
- **Data:** historical 10-K text from EDGAR (full history is available, dated by filing date → no
  look-ahead if you enter *after* the filing date). Cost: bulk large-doc retrieval, SEC-rate-limited
  (~10 req/s, ~0.15s/doc here) → a few hundred names × ~8 annual filings = a multi-hour data job.
  Cache aggressively.
- **Method:** per filing season, rank the universe by `change_score`; form quintiles; measure forward
  **21- and 63-day** returns (excess vs SPY), walk-forward by year, with fold consistency + tail —
  same statistical guards as `signal_direction_matrix.py`.
- **Bounded first pass:** 60–80 names, last ~5 years, 21/63d horizon — enough to see whether the
  most-changed quintile underperforms before committing to the full bulk fetch.
- **Caveat:** survivorship (today's listings); cosine on full text is a proxy — risk-factor-section
  (Item 1A) or Jaccard refinements may sharpen it.

## 2. Forms of rich unstructured context

### The binding constraint first
The daily decision prompt is already **~379k tokens / ~$1.18**. A single 10-K is 50–150k tokens. So
"feed the LLM rich context" can NOT mean dumping raw text. Every form below has to be **distilled**
into something the decision prompt can afford, via one of three patterns:

- **(F) Feature extraction** — compute a score/flag from the text, feed the number (e.g.
  `filing_similarity` → `change_score`). Cheapest, most testable, no extra LLM cost. Default choice.
- **(S) LLM pre-summary** — a cheap model (Haiku) distills the text to a few-hundred-token summary
  appended to the candidate's snapshot. Use when the *content* matters, not just a score.
- **(R) Retrieval** — feed only the most relevant snippet (e.g. the changed risk-factor paragraph).
  Use when a specific passage is the signal.

### The forms, best-fit first
| Form | What it captures | Source (cost) | Distill | Look-ahead risk |
|---|---|---|---|---|
| **10-K/10-Q text & diffs** | Lazy Prices change; risk-factor expansion; going-concern language | EDGAR (free) | F (+R for the changed section) | Low — use filing date |
| **Filing micro-events** | NT 10-K (late filing), 8-K Item 4.01 auditor change, restatement | EDGAR (free) | F (flags) | Low |
| **8-K full narrative** | *why* behind the event flags we already extract | EDGAR (free) | S | Low |
| **Earnings-call transcripts** | management tone, hedging, Q&A evasion → post-call drift | paid/semi-free | S + F (tone score) | Med — use call timestamp |
| **News full-text + novelty** | is the news *new* vs rehash; cross-article stance | AV/yfinance (free-ish) | S + F (novelty/sentiment) | **High** — publication-time discipline |
| **Analyst rationale** | reasoning behind a rating/▲▼, not just the number | limited | S | Med |
| **Social/forum (Reddit/StockTwits)** | retail attention spike + sentiment | free-ish, noisy | F (attention/sentiment) | Med + squeeze risk |
| **Alt-text (jobs, Glassdoor, patents)** | slow fundamental drift | hard/paid | F | Low but slow |

### Recommended architecture: a "context card" pre-pass
Rather than bolt each source onto the prompt, run a **distillation pre-pass** per candidate that emits
a compact **context card** (a few hundred tokens): `filing_change_score`, most-recent-material-8K
one-line summary, news-novelty + stance, transcript-tone (when available). Append the card to the
snapshot. This (a) keeps the decision prompt affordable, (b) is the single mechanism that delivers
Lazy Prices *and* every other text form, and (c) is exactly the input that lets the LLM's **synthesis**
(the experiment's actual thesis) separate from a deterministic screen — it can weigh a filing change
*against* the news *against* the fundamentals, which no single column encodes.

**Discipline:** any context card changes the decision inputs → it's an experiment arm, freeze-gated,
and every text form needs strict publication-timestamp handling to avoid look-ahead (the failure mode
that has bitten us before). Start with the all-free, low-look-ahead, feature-extraction forms (filing
diffs + filing micro-events), prove the card pipeline, then add summaries (8-K, transcripts).
