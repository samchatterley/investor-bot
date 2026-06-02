# InvestorBotHard — Claude Code Instructions

## Session start

If `TODO.md` exists in the project root, read it and restore the items into TodoWrite before doing anything else. This is the persistent backlog — do not skip this step.

## Todo hygiene

- Mark tasks completed in TodoWrite as soon as they are done, not in a batch at the end.
- TODO.md is auto-written by a PostToolUse hook on every TodoWrite call — do not edit it manually.

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
