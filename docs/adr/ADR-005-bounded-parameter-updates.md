# ADR-005: Bounded Parameter Updates

**Date:** April 2026
**Status:** Accepted

## Context

InvestorBot's operating behaviour is governed by a set of configuration parameters — confidence thresholds, stop-loss percentages, profit-taking targets, and maximum hold durations. These parameters were chosen at system design time but market conditions change: volatility regimes shift, the optimal trailing stop for a low-volatility environment is different from the right value in a high-volatility one, and a system that cannot adapt will degrade in performance over time.

Two unsatisfying alternatives were considered. A fully static configuration means the system cannot adapt at all; human operators must manually review and update parameters, which in practice means they rarely get updated. A fully autonomous self-modification loop — where Claude can propose and apply any configuration change — allows unbounded parameter drift and creates a feedback loop where the system could progressively widen its own risk envelope in response to short-run performance, which is unsafe.

A constrained middle ground is needed: the system should be able to propose evidence-based parameter adjustments, but the adjustable range for each parameter must be fixed by humans and enforced deterministically, and all proposed changes must be visible to human operators before taking effect.

## Decision

The weekly self-review process (run every Sunday) allows Claude to propose changes to four specific operating parameters based on the past week's trade outcomes. The proposal is expressed as a structured output specifying a new value for each parameter. Before any value is written to configuration, the proposed value is validated against hard-coded bounds defined in source code:

| Parameter | Minimum | Maximum |
|---|---|---|
| `MIN_CONFIDENCE` | 6 | 9 |
| `TRAILING_STOP_PCT` | 2% | 8% |
| `PARTIAL_PROFIT_PCT` | 5% | 20% |
| `MAX_HOLD_DAYS` | 2 | 7 |

These bounds are enforced in the validation layer, not in the prompt. Claude is told what the bounds are so it reasons within them, but the bounds are also enforced independently in code — Claude's stated value is clamped and rejected if it falls outside the range, regardless of the reasoning provided.

The configuration file is written atomically: either all proposed values that pass validation are written in a single operation, or none are. Partial writes (some parameters updated, others not) do not occur.

All proposed values — both accepted and rejected — are included in the Sunday summary email sent to the operator, along with Claude's evidence-based reasoning for each proposal. The operator can review the changes and, if concerned, revert by editing the config file or triggering the halt switch.

## Consequences

**Positive:**
- The system can adapt to changing market conditions without requiring manual operator intervention each week, reducing operational overhead while maintaining safety.
- Hard-coded bounds mean Claude cannot propose a value that pushes the system outside the pre-approved operating envelope, no matter what reasoning it provides. The bounds are a human decision encoded in source code, not a prompt instruction.
- Atomic writes prevent a class of bug where a crash during configuration update leaves the system in an inconsistent state with some parameters at new values and others at old values.
- Full transparency: every proposed change and its reasoning appears in the weekly email before it takes effect, giving operators a regular review touchpoint.

**Negative:**
- The four adjustable parameters and their bounds represent a human judgement call made at design time. If market conditions require a value outside the coded bounds (e.g. a volatility event where a trailing stop wider than 8% is appropriate), a human must change the source code — the system cannot adapt beyond its envelope.
- The weekly cadence means the system is always one week behind in adapting to rapidly changing conditions. A parameter that becomes suboptimal on Monday is not updated until Sunday at the earliest.
- The Sunday email requires an engaged operator to be meaningful. If the email is ignored, the oversight value of the transparency mechanism is lost.
