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
