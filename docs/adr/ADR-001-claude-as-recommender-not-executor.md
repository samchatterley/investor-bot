# ADR-001: Claude as Recommender, Not Executor

**Date:** April 2026
**Status:** Accepted

## Context

Two design poles were considered for InvestorBot's architecture. At one extreme, a fully autonomous AI trading system where Claude has direct access to the Alpaca API and can place, modify, and cancel orders without intermediary checks. At the other extreme, a fully manual rules-based system where a deterministic quant model drives all decisions with no LLM involvement.

A fully autonomous AI executor raises serious concerns: LLM outputs are non-deterministic, difficult to formally verify, and can hallucinate or be manipulated via prompt injection. Giving Claude direct brokerage access would mean a single bad output could execute a harmful trade with no interception point. Conversely, a purely rules-based system lacks the contextual reasoning ability to weigh qualitative signals (news sentiment, earnings context, macro narrative) against quantitative ones.

A hybrid architecture was chosen: Claude performs the reasoning and produces a structured recommendation, but it has no ability to act on that recommendation directly.

## Decision

Claude generates trade recommendations expressed as structured output (ticker, direction, confidence score, position size, rationale). A deterministic validator and multi-layer risk gate sit between Claude's output and any order placement. Claude cannot call the Alpaca API, cannot read live account balances, cannot modify system configuration, and cannot directly influence execution in any way other than producing a recommendation object that downstream components may accept, reject, or ignore.

The execution pipeline is strictly one-directional: Claude → validator → risk gate → Alpaca API. No feedback channel from execution back to Claude exists within a single run.

## Consequences

**Positive:**
- Every trade decision produces an interpretable audit trail: Claude's recommendation and reasoning are logged before any execution decision is made, making post-hoc review straightforward.
- Blast radius is bounded. A hallucinated or adversarially influenced Claude output cannot bypass the validator and risk gate; the worst case is a rejected recommendation, not an erroneous trade.
- Constraints and overrides are easy to apply at the risk gate layer without touching the LLM or its prompts.
- The system is easy to explain to regulators, auditors, or stakeholders — the AI advises, humans (via configured rules) decide.

**Negative:**
- Higher latency than a pure quant system. Each run incurs an Anthropic API call, which adds hundreds of milliseconds to seconds of round-trip time.
- Ongoing API cost per run. Every scheduled invocation consumes tokens regardless of whether a trade is ultimately placed.
- The quality of recommendations is bounded by Claude's context window and the quality of data fed to it; subtle quantitative signals that would be trivial in a pure quant system require careful prompt engineering to convey.
