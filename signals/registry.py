"""Single source of truth for signal name sets used across the codebase.

All places that need to know "what signals can fire?", "what can the AI cite?",
or "is this signal valid?" should import from here rather than maintaining
independent copies.

Derives sets from signals.evaluator so there is exactly one place to add or
remove a signal name.
"""

from signals.evaluator import GLOBALLY_DISABLED, SIGNAL_PRIORITY

# All long signals that exist in the evaluator, minus globally-disabled ones.
ACTIVE_LONG_SIGNALS: frozenset[str] = frozenset(SIGNAL_PRIORITY.keys()) - GLOBALLY_DISABLED

# Signals the AI may cite as key_signal.  Equals the active set plus the
# "unknown" catch-all (used when no specific signal is the primary driver).
# Disabled signals are intentionally excluded: offering them to the AI wastes
# prompt tokens and can cause the AI to cite a signal that will never fire,
# tainting the empirical win-rate stats that feed back into position sizing.
AI_CITEABLE_SIGNALS: frozenset[str] = ACTIVE_LONG_SIGNALS | {"unknown"}
