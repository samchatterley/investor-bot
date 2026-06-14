"""Lazy-loaded FinBERT sentiment classifier for financial text.

Wraps ProsusAI/finbert via the HuggingFace transformers pipeline.
Returns None gracefully when the model is unavailable or input is too short.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# None  → not yet attempted; False → unavailable (load failed)
_pipeline: Any = None

_MODEL = "ProsusAI/finbert"
_MIN_TEXT_LEN = 20
_MAX_TEXT_LEN = 2000
_LABELS = ("positive", "negative", "neutral")


def _load_pipeline() -> None:
    """Attempt to import transformers and load the FinBERT pipeline.

    Sets _pipeline to the loaded pipeline on success, or False on any failure.
    Never raises.
    """
    global _pipeline
    try:
        from transformers import pipeline as _hf_pipeline

        _pipeline = _hf_pipeline(
            "text-classification",
            model=_MODEL,
            return_all_scores=True,
        )
    except Exception as exc:  # includes ImportError, OSError, etc.
        logger.warning("finbert: pipeline unavailable: %s", exc)
        _pipeline = False


def is_available() -> bool:
    """Return True if the FinBERT pipeline is loaded and ready.

    Triggers the first load attempt if not yet tried.
    """
    global _pipeline
    if _pipeline is None:
        _load_pipeline()
    return _pipeline is not False


def classify_text(text: str) -> dict | None:
    """Classify a single text snippet with FinBERT.

    Returns a dict with keys:
        label    — "positive" | "negative" | "neutral"
        score    — confidence of the winning label (normalised)
        positive — normalised positive probability
        negative — normalised negative probability
        neutral  — normalised neutral probability

    Returns None when text is too short, pipeline unavailable, or any error.
    """
    global _pipeline
    if len(text.strip()) < _MIN_TEXT_LEN:
        return None

    if _pipeline is None:
        _load_pipeline()
    if _pipeline is False:
        return None

    try:
        raw = _pipeline(text[:_MAX_TEXT_LEN])
        # raw is [[{"label": ..., "score": ...}, ...]]
        scores_list: list[dict] = raw[0]
        total = sum(item["score"] for item in scores_list) or 1.0
        scores = {item["label"].lower(): item["score"] / total for item in scores_list}

        # Ensure all three labels are present (defensive)
        for lbl in _LABELS:
            scores.setdefault(lbl, 0.0)

        best_label = max(scores, key=lambda k: scores[k])
        return {
            "label": best_label,
            "score": scores[best_label],
            "positive": scores["positive"],
            "negative": scores["negative"],
            "neutral": scores["neutral"],
        }
    except Exception as exc:
        logger.warning("finbert: classify_text error: %s", exc)
        return None


def classify_texts(texts: list[str]) -> list[dict | None]:
    """Classify a list of texts, returning one result per input.

    Each element is the result of classify_text; None for failures.
    """
    return [classify_text(t) for t in texts]
