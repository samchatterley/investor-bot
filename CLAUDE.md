# InvestorBotHard — Claude Code Instructions

## Session start

If `TODO.md` exists in the project root, read it and restore the items into TodoWrite before doing anything else. This is the persistent backlog — do not skip this step.

## Todo hygiene

- Mark tasks completed in TodoWrite as soon as they are done, not in a batch at the end.
- **Never mark items completed unless the work is fully shipped** (code committed, tests green). Do not mark pending items completed at session end just to tidy up — leave them pending.
- TODO.md is auto-written by a PostToolUse hook on every TodoWrite call — do not edit it manually.
- The hook preserves the file if the pending list would become empty — accidental batch-completes are ignored. A pre-commit git hook also blocks staging TODO.md with fewer than 3 pending items.

## After every push

Restart the scheduler immediately after every `git push` — unless market hours are active.

Restart command:
```
python scripts/run_scheduler.py
```

Run this in the `investorbot` tmux session. Do **not** use `python main.py` (that starts open mode, not the scheduler).

**Never restart Mon–Fri 09:30–16:00 ET (14:30–21:00 BST).** If market hours are active, defer the restart until after 16:00 ET and note it as pending.

**Always kill the existing process (Ctrl-C) before restarting.** The old process keeps running otherwise.

## Scheduler schedule

| ET | BST | Job |
|---|---|---|
| 07:00 | 12:00 | pre-market data prefetch (warms cache, no trading) |
| 09:31 | 14:31 | open_sells |
| 10:00 | 15:00 | open_buys |
| 12:00 | 17:00 | midday |
| 15:30 | 20:30 | close |

## Disabling a signal — checklist

`evaluate_signals()` and `evaluate_short_signals()` merge `GLOBALLY_DISABLED` / `SHORT_GLOBALLY_DISABLED` into the blocked set, so adding a signal to one of those frozensets silently makes its detection branch unreachable. Every disable **must** do all of the following in the **same commit**, or the suite breaks (this is how v1.98 shipped 7 failing tests):

1. Add the signal name to `GLOBALLY_DISABLED` / `SHORT_GLOBALLY_DISABLED` with a one-line evidence comment (trades, WR, avg, ΔSharpe).
2. Add `# pragma: no cover — <signal> in GLOBALLY_DISABLED` to the `matched.append("<signal>")` line — the branch can never execute now.
3. **Update its tests.** Find every `assert "<signal>" in signals` / `assertIn("<signal>", ...)` / `assertEqual(_entry_signal(...), "<signal>")` and convert to the disabled pattern: assert the signal is in the disabled frozenset **and** does not fire when its conditions are met. `grep -rn "<signal>" tests/` to find them all.
4. Run the affected test files (`test_backtest.py`, `test_new_signals.py`, `test_short_side.py`, `test_wiring.py` — at minimum the ones that reference the signal) before staging.
5. Update `docs/signals.md` (move to disabled table), `README.md` (active/disabled counts), and `CHANGELOG.md`.
6. If the signal appeared in `SYSTEM_PROMPT` (`analysis/ai_analyst.py`), remove it — the wiring parity tests in `test_wiring.py` enforce this for long signals.
