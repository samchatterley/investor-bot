"""Detect index inclusion/deletion (the index_change material-context category) from news headlines.

Index membership changes (S&P 500/400/600, Nasdaq-100, Dow Jones, Russell) are announced by the index
providers, which have no clean point-in-time API. But the events are reliably newsworthy, so we detect
them from the headlines the bot already fetches (data/news_fetcher.py). This is a direction-agnostic
material-context enrichment flag — it only marks that a membership change is being reported; the AI
judges the implication.

Precision over recall: a bare "added"/"S&P 500" mention is not enough (headlines say "the S&P 500 rose"
constantly). A match requires membership phrasing — a change verb plus a preposition next to a named
index ("added to the S&P 500", "removed from the Nasdaq-100", "to replace X in the Dow Jones") or the
explicit "<index> inclusion/addition/deletion" noun. Negative tests lock in the generic-mention cases.
"""

from __future__ import annotations

import re

_INDEX = (
    r"(?:s&p\s*(?:500|400|600)|nasdaq[\- ]?100|dow jones(?:\s+industrial average)?|"
    r"russell\s*(?:1000|2000|3000))"
)

# A membership change: an add/remove verb bound to a preposition next to a named index, a
# replacement within an index, or the explicit inclusion/addition/deletion noun.
_INDEX_CHANGE = re.compile(
    r"(?:"
    r"(?:added|to be added|set to join|will join|joins?|joining|to join)\s+(?:to\s+)?(?:the\s+)?"
    + _INDEX
    + r"|(?:removed|dropped|deleted|ousted)\s+from\s+(?:the\s+)?"
    + _INDEX
    + r"|replace[ds]?\s+[\w\s.&,'-]{0,30}\bin\s+(?:the\s+)?"
    + _INDEX
    + r"|"
    + _INDEX
    + r"\s+(?:inclusion|addition|deletion)"
    + r"|index\s+(?:inclusion|addition|deletion)"
    r")",
    re.I,
)


def classify_index_change(headlines: list[str]) -> dict | None:
    """Return ``{"detected": True, "headline": <matched headline>}`` if any headline reports an index
    membership change, else None."""
    for headline in headlines:
        if _INDEX_CHANGE.search(headline):
            return {"detected": True, "headline": headline}
    return None
