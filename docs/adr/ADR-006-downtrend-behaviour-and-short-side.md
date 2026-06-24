# ADR-006 — Downtrend behaviour: stop long-churn, make the short side first-class

- **Status:** Accepted (2026-06-24)
- **Supersedes/relates:** ADR-001 (Claude as recommender, not executor)
- **Trigger:** Live paper account bled ~3% (100k → ~97k) over 06-18…06-24 with **zero shorts**
  despite a `DEFENSIVE_DOWNTREND` regime — operator-reported, then root-caused here.

## Context

A multi-day investigation of the live logs + DB established two things.

### 1. The bot was churning longs against its own regime rules

The entry gate and the exit gate hold **opposite views of the same regime**:

- **Entry:** `skip_buys` (main.py) blocks buys on `mc.regime.is_bearish`, which is **False** for
  `DEFENSIVE_DOWNTREND`. The regime policy compounds it: `DEFENSIVE_DOWNTREND` is
  `block_new_buys=False, max_orders_per_run=2` — it *actively permits* 2 new longs per run.
- **Exit:** the regime-change exit (main.py, `_DEFENSIVE_REGIMES = {DEFENSIVE_DOWNTREND, BEAR_MARKET}`)
  force-closes any long with `days_held < 2`.

So the bot **opens longs it is structurally committed to dumping within a day** — sometimes the same
day (e.g. `BUY CINF 17:05` → `Regime-change exit: CINF 20:33`, both 06-24). Evidence: 15 trades since
06-18, **27% win rate, −0.89%/trade**, almost all held 1 day, exiting via the regime change — *against
the AI's explicit HOLD* on nearly every name ("Hold — give the BB squeeze time", "Holding is
warranted"). The `ai_sell` exit-reason mislabel (see the attribution bug, separate fix) hid that these
were regime exits, not AI sells.

### 2. The bot has no working downside tool, so churn is its only "action"

- **Single-name shorts** are blocked: `_execute_shorts` checks VIX-term-structure inversion
  (`VIX9D/VIX > 1.05`) **first and returns early**, overriding the standalone-short permission that
  bear regimes otherwise grant. A grind-down (VIX ~19–20, not inverted) never clears it.
- The **trend/technical short book is disabled** — 17 signals in `SHORT_GLOBALLY_DISABLED`
  (`ema_breakdown`, `death_cross`, `rs_deterioration`, `overbought_downtrend`, …) for no backtest
  edge. The live short book is a handful of sparse event signals.
- **Shorts bypass the AI entirely.** Longs run through `ai_analyst` (news/sentiment/context/
  conviction); shorts are a mechanical `scan_short_candidates` + rule gates. This is an asymmetry, not
  a design intent.
- The **index hedge** (the intended downtrend tool) is non-functional: `Index hedge: $10 < 1 share of
  SPY @ $400.00` (sized to zero shares) and `Account safety check failed: Broker account has
  PDT/margin flag`.

With no working short and no working hedge, the bot's only move in a downtrend is to keep buying longs
— which it then regime-dumps. That is the bleed.

## Decision

**A. Stop the churn now (this ADR's code change).** Make entry agree with the exit: in
`_DEFENSIVE_REGIMES`, do not open new longs. Concretely, set the `DEFENSIVE_DOWNTREND` regime policy to
`block_new_buys=True, max_orders_per_run=0` — making it identical to the existing `STRESS_RISK_OFF` /
`UNKNOWN` no-buy regimes, which already block entries via the `max_orders_per_run=0` → `effective_max_orders`
cap (an already-tested path, no new branch). Rationale: if a regime is bearish enough to dump fresh
longs, it is bearish enough not to open them. The regime-change *exit* is left intact — it correctly
protects positions caught by a regime *flip* (a NEUTRAL_CHOP long that becomes defensive). This is an
**entry-side** fix only.

*Note:* `block_new_buys` is currently only half-wired — the buy gate enforces via `max_orders_per_run=0`,
not the flag. Honouring the flag in `skip_buys` for clearer "blocks new entries" logging is a separate
follow-up cleanup, deliberately kept out of this urgent fix to minimise blast radius on the live buy path.

**Accepted consequence:** in `DEFENSIVE_DOWNTREND` the bot will now sit mostly in **cash** (no new
longs, no working shorts) until the short side is rebuilt. Cash is a position; it is strictly better
than churning longs at a loss plus costs.

**B. Make the short side first-class (forward scope; built after A).** Five levers, in priority:

1. **Regime-first short gate** — VIX-inversion becomes an edge *modifier* (size up when inverted), not
   a hard precondition. A confirmed bear regime + weak breadth should permit shorts without a vol panic.
2. **Route shorts through the AI** — short candidates get the same context packet (news, filings,
   sentiment, conviction) and the AI selects/vetoes them exactly like longs. Dovetails with the
   experiment arm-pass plumbing (same "AI sees per-candidate context" work).
3. **Fix the index hedge** — sizing floor + the PDT/margin flag — as the immediate, reactive downtrend
   protection while the single-name redesign lands.
4. **Reconsider the trend-short book** — the disabled signals failed backtest, so "reactivity" may be
   better served by the ETF/index hedge + event shorts than by resurrecting them. Needs its own
   backtest pass before re-enabling anything (per the disable-checklist in CLAUDE.md).
5. **Defensive-long policy** — revisit whether *any* longs (e.g. genuine defensive/gold catalysts)
   should be allowed in `DEFENSIVE_DOWNTREND` and, if so, be exempted from the `days_held<2` dump.
   Deferred until the short side works.

## Consequences

- Immediate: the daily long-churn bleed stops; downtrend = cash until B lands.
- The short-side redesign (B) is a multi-step build; B2 (AI-driven shorts) is the largest and is
  sequenced with the experiment arm-pass work.
- Experiment integrity: the `ai_sell` mislabel that masked the regime exits is fixed separately (the
  exit-attribution fix); both are freeze-blockers for clean sell-side data.
