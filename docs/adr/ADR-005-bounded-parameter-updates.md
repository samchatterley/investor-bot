# ADR-005: Bounded Parameter Updates

**Date:** April 2026
**Status:** Accepted (updated May 2026)

## Context

InvestorBot's operating behaviour is governed by a set of configuration parameters — confidence thresholds, stop-loss percentages, profit-taking targets, and maximum hold durations. These parameters were chosen at system design time but market conditions change: volatility regimes shift, the optimal trailing stop for a low-volatility environment is different from the right value in a high-volatility one, and a system that cannot adapt will degrade in performance over time.

Two unsatisfying alternatives were considered. A fully static configuration means the system cannot adapt at all; human operators must manually review and update parameters, which in practice means they rarely get updated. A fully autonomous self-modification loop — where Claude can propose and apply any configuration change — allows unbounded parameter drift and creates a feedback loop where the system could progressively widen its own risk envelope in response to short-run performance, which is unsafe.

A constrained middle ground is needed: the system should be able to propose evidence-based parameter adjustments, but every proposal must be reviewed and applied manually by the operator, and the adjustable range for each parameter must be fixed by humans and enforced deterministically.

## Decision

The weekly self-review process (run every Sunday) allows Claude to propose changes to four specific operating parameters based on the past week's trade outcomes. The proposal is expressed as a structured output specifying a new value for each parameter. Proposed values are validated against hard-coded bounds defined in source code:

| Parameter | Minimum | Maximum |
|---|---|---|
| `MIN_CONFIDENCE` | 6 | 9 |
| `TRAILING_STOP_PCT` | 2% | 8% |
| `PARTIAL_PROFIT_PCT` | 5% | 20% |
| `MAX_HOLD_DAYS` | 2 | 7 |

These bounds are enforced in the validation layer, not in the prompt. Claude is told what the bounds are so it reasons within them, but the bounds are also enforced independently in code — Claude's stated value is rejected if it falls outside the range, regardless of the reasoning provided.

**Configuration is not written automatically.** Proposals are included in the Sunday summary email to the operator, along with Claude's evidence-based reasoning for each suggestion. The operator reviews the proposals and manually edits `.env` or `config.py` to apply any changes they accept. The bot takes no automated write action on configuration files.

All proposed values — both in-bounds and out-of-bounds — appear in the email. Out-of-bounds proposals are flagged with the violated constraint so the operator understands why they were not actioned.

## Consequences

**Positive:**
- The system surfaces evidence-based adaptation recommendations without writing configuration autonomously. The human operator remains the only agent that can change operating parameters.
- Hard-coded bounds mean Claude cannot propose a value that pushes the system outside the pre-approved operating envelope, no matter what reasoning it provides. The bounds are a human decision encoded in source code, not a prompt instruction.
- Full transparency: every proposed change and its reasoning appears in the weekly email as a review artefact.
- No risk of a bug in an atomic-write path corrupting `.env` or `config.py`.

**Negative:**
- The four adjustable parameters and their bounds represent a human judgement call made at design time. If market conditions require a value outside the coded bounds (e.g. a volatility event where a trailing stop wider than 8% is appropriate), a human must change the source code — the system cannot adapt beyond its envelope.
- The weekly cadence means the system is always one week behind in adapting to rapidly changing conditions. A parameter that becomes suboptimal on Monday is not updated until Sunday at the earliest, and only if the operator acts on the proposal.
- The Sunday email requires an engaged operator to be meaningful. If the email is ignored, no adaptation occurs.
- Manual application of proposals creates a risk that operators apply only partial changes, introducing inconsistency. The email is formatted to make this obvious.
