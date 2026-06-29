"""Lazy Prices signal — textual similarity between a company's consecutive periodic filings.

Cohen, Malloy & Nguyen (2020), "Lazy Prices": firms that materially CHANGE their 10-K/10-Q language
year-over-year subsequently underperform; copy-paste ("lazy") filers do fine. So a LOW similarity
between a company's two most recent same-form filings is a bearish change flag; HIGH similarity is
benign. The effect is strongest in the risk-factor / MD&A narrative.

Reuses the EDGAR plumbing in data/edgar_client (CIK lookup, submissions, doc-text fetch). This is the
signal primitive only — it is NOT yet wired into trading. Two things to settle first (see
docs/lazy_prices_and_context.md): (1) calibrate the bearish threshold from a backtest, and (2) the
horizon mismatch — Lazy Prices is a slow (multi-month) drift, so it likely fits as a risk-flag /
slow tilt rather than a 1-5 day swing entry.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter

from data import edgar_client as _ec

logger = logging.getLogger(__name__)

# Enough chars to span risk factors + MD&A (a 10-K narrative); _strip_html strips before truncating.
_FILING_MAX_CHARS = 200_000
_LOOKBACK_DAYS = 800  # spans two annual 10-Ks
# Provisional — to be calibrated from the backtest. Below this similarity = notable change = bearish.
_SIM_BEARISH_THRESHOLD = 0.80

_TOKEN_RE = re.compile(r"[a-z]{3,}")  # words ≥3 letters (text is already lowercased by _strip_html)
# Common filler that adds match-noise without signal — a small stoplist, not a full NLP pipeline.
_STOP = frozenset(
    {
        "the",
        "and",
        "that",
        "for",
        "with",
        "this",
        "from",
        "our",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "its",
        "such",
        "which",
        "other",
        "may",
        "not",
        "but",
        "will",
        "would",
        "any",
        "all",
        "been",
        "under",
        "than",
        "into",
        "upon",
        "also",
        "these",
        "those",
        "there",
    }
)


def _tokens(text: str) -> Counter:
    """Term-frequency counts of meaningful tokens in `text`."""
    return Counter(t for t in _TOKEN_RE.findall(text) if t not in _STOP)


def cosine_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity (0..1) of the two texts' term-frequency vectors. 0 if either is empty."""
    ca, cb = _tokens(text_a), _tokens(text_b)
    if not ca or not cb:  # empty (or all-stopword) → undefined similarity
        return 0.0
    # ca/cb are non-empty term counts, so both norms are strictly positive here.
    dot = sum(ca[t] * cb[t] for t in (ca.keys() & cb.keys()))
    norm_a = math.sqrt(sum(v * v for v in ca.values()))
    norm_b = math.sqrt(sum(v * v for v in cb.values()))
    return dot / (norm_a * norm_b)


def get_filing_change(symbol: str, form: str = "10-K") -> dict | None:
    """Compare the two most recent `form` filings for `symbol`. Returns the change record or None.

    Record: {symbol, form, latest_date, prior_date, similarity, change_score, lazy_prices_bearish}.
    None when the CIK is unknown, fewer than two comparable filings exist, or text can't be fetched.
    Default form="10-K" gives a clean year-over-year comparison; 10-Q would need same-quarter
    alignment to avoid seasonality (a refinement, see the scoping doc).
    """
    cik = _ec._get_cik_map().get(symbol.upper())
    if not cik:
        return None

    filings = _ec._get_recent_filings(cik, [form], _LOOKBACK_DAYS)
    if len(filings) < 2:
        return None
    latest, prior = filings[0], filings[1]

    text_latest = _ec._fetch_filing_text(cik, latest["accession"], latest["doc"], _FILING_MAX_CHARS)
    text_prior = _ec._fetch_filing_text(cik, prior["accession"], prior["doc"], _FILING_MAX_CHARS)
    if not text_latest or not text_prior:
        return None

    sim = cosine_similarity(text_latest, text_prior)
    return {
        "symbol": symbol.upper(),
        "form": form,
        "latest_date": latest["filing_date"],
        "prior_date": prior["filing_date"],
        "similarity": round(sim, 4),
        "change_score": round(1.0 - sim, 4),
        "lazy_prices_bearish": sim < _SIM_BEARISH_THRESHOLD,
    }
