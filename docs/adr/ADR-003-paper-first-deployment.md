# ADR-003: Paper-First Deployment

**Date:** April 2026
**Status:** Accepted

## Context

InvestorBot is an autonomous system that, once started, places orders on a schedule without per-trade human approval. This architecture is inherently high-blast-radius: a misconfigured parameter, an API regression, a data feed outage returning bad values, or an unexpected market event can cause large real financial losses very quickly and with no human in the loop to intervene.

Unlike a manual trading tool where a user reviews each order before submission, an autonomous system requires the safety envelope to be defined in advance and enforced by the system itself. The default posture of the system must reflect this asymmetry: the cost of a false positive (paper trade that would have been profitable) is an opportunity cost, while the cost of a false negative (real trade executed under unsafe conditions) can be an immediate and irreversible monetary loss.

## Decision

Paper trading is the default operating mode for all deployments. The default `.env.example` configuration points to Alpaca's paper trading endpoint. Switching to live trading requires explicit opt-in through multiple independent safeguards that must all be satisfied simultaneously:

1. The `LIVE_CONFIRM` environment variable must be set to a specific confirmation string (not just `true` or `1`).
2. The Alpaca API credentials in the environment must be live-account credentials (the system validates the account type returned by the Alpaca API on startup).
3. No `HALT_FILE` must be present on disk at the configured path.

In addition, the following controls are active in all modes but are the primary protection layer in live mode:

- **HALT_FILE kill switch:** The presence of a file at the configured `HALT_FILE` path causes the scheduler to stop processing immediately without placing any orders.
- **Daily loss limit with auto-liquidation:** If realised-plus-unrealised losses for the calendar day exceed `MAX_DAILY_LOSS_PCT` of starting equity, all open positions are liquidated and no new orders are placed until the next trading day.
- **Circuit breaker:** If the number of consecutive rejected recommendations exceeds a threshold, the system pauses and sends an alert rather than continuing to retry.
- **Fat-finger guard:** Any single order whose notional value exceeds `FAT_FINGER_MAX_NOTIONAL` is rejected at the risk gate regardless of Claude's recommendation.
- **Daily notional cap:** The total notional value of all orders placed in a single calendar day is capped at `MAX_DAILY_NOTIONAL`.

## Consequences

**Positive:**
- New deployments are safe by default. An operator must take deliberate, multi-step action to enable live trading, making accidental live exposure effectively impossible.
- The paper trading environment provides a realistic rehearsal space where configuration errors, data feed issues, and unexpected Claude behaviours can be observed and corrected without financial cost.
- The layered safeguards (halt file, loss limit, circuit breaker, fat-finger guard, notional cap) provide defence in depth once live mode is enabled — no single point of failure can cause unconstrained losses.
- Operators can quickly halt the system in an emergency by creating the halt file, without needing to kill the process or modify configuration.

**Negative:**
- Paper trading results do not perfectly reflect live execution due to fill simulation differences; a strategy that looks good in paper mode may perform differently in live mode due to slippage and partial fills.
- The multi-step live opt-in adds friction that may frustrate experienced operators who understand the risks and want to move to live trading quickly.
- The daily loss limit with auto-liquidation may crystallise losses in fast-moving markets where holding a position would recover value intraday.
