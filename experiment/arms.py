"""Arm 2 / Arm 3 prompt contracts for the contextual-value experiment.

Implements the two LLM arms from docs/EXPERIMENT.md section 2.1 and the output contract
in section 7:

  Arm 2 (structured-only): reasons over a fixed, mechanically-rendered prose view of the
    structured evidence and emits a structured_conviction (1-10) plus reasoning.
  Arm 3 (contextual): receives Arm 2's EXACT prompt plus an appended block carrying Arm 2's
    frozen output and the context packet, and emits an explicit, coarse context_adjustment.

Isolation constraint: Arm 3's prompt is Arm 2's exact prompt plus an appended context block;
nothing in the base prompt differs. The instrument starts COARSE (negative/neutral/positive
plus veto); granularity is only refined if the noise audit (experiment/noise_audit.py) supports
it. final_conviction is a mechanical sum, never a free-form re-derivation.

These functions are pure (no network). The live LLM caller is wired in
scripts/phase0_noise_audit.py.
"""

from __future__ import annotations

# Coarse adjustment scale (start coarse per the noise-audit decision).
COARSE_ADJUSTMENTS: tuple[str, ...] = ("negative", "neutral", "positive")
ADJUSTMENT_VALUE: dict[str, int] = {"negative": -1, "neutral": 0, "positive": 1}
REASON_CODES: tuple[str, ...] = (
    "news",
    "filing",
    "earnings",
    "macro",
    "sector",
    "options",
    "other",
)

_CONVICTION_MIN = 1
_CONVICTION_MAX = 10

# Fixed, ordered structured-evidence template (the versioned mechanical prose view). Only keys
# present on a candidate are rendered, always in this order, so the prose is reproducible.
_FIELD_ORDER: tuple[tuple[str, str], ...] = (
    ("symbol", "Symbol"),
    ("evidence_score", "Evidence score"),
    ("regime", "Market regime"),
    ("rsi_14", "RSI(14)"),
    ("macd_diff", "MACD diff"),
    ("bb_pct", "Bollinger %B"),
    ("vol_ratio", "Volume ratio"),
    ("price_vs_ema21_pct", "Price vs EMA21 %"),
    ("rel_strength_5d", "Relative strength 5d"),
    ("rel_strength_10d", "Relative strength 10d"),
    ("price_vs_52w_high_pct", "Price vs 52w high %"),
)


def build_structured_prose(candidate: dict) -> str:
    """Render a candidate's structured features as fixed, ordered prose (Arm 2/Arm 3 input).

    Mechanical and versioned: iterates _FIELD_ORDER and includes only present, non-None fields.
    Matched signals, if present, are appended as a final line.
    """
    lines = []
    for key, label in _FIELD_ORDER:
        value = candidate.get(key)
        if value is not None:
            lines.append(f"- {label}: {value}")
    signals = candidate.get("matched_signals") or candidate.get("signals")
    if signals:
        lines.append(f"- Fired signals: {', '.join(signals)}")
    return "\n".join(lines)


def build_arm2_prompt(structured_prose: str) -> str:
    """Arm 2 base prompt: structured evidence only, no context."""
    return (
        "You are a short-term US equities analyst scoring a single pre-filtered candidate using "
        "only the structured evidence below. Do not speculate about news, filings, or events you "
        "are not given.\n\n"
        "STRUCTURED EVIDENCE:\n"
        f"{structured_prose}\n\n"
        "Return a structured_conviction from 1 (avoid) to 10 (highest quality), a structured_rank "
        "hint, and a one-line structured_reasoning grounded in the values above."
    )


def build_arm3_prompt(structured_prose: str, arm2_output: dict, context_block: str) -> str:
    """Arm 3 prompt: Arm 2's exact prompt plus an appended context-and-revision block.

    The base (Arm 2) prompt is byte-identical; only the appended block adds Arm 2's frozen
    output, the context packet, and the coarse-revision instruction.
    """
    base = build_arm2_prompt(structured_prose)
    conviction = arm2_output.get("structured_conviction", "")
    reasoning = arm2_output.get("structured_reasoning", "")
    return (
        f"{base}\n\n"
        "APPENDED CONTEXT REVISION:\n"
        f"Your structured-only view was structured_conviction={conviction} "
        f'(reasoning: "{reasoning}").\n\n'
        "TIMESTAMP-SAFE CONTEXT:\n"
        f"{context_block or '(no admissible context)'}\n\n"
        "Given ONLY what this context adds or removes relative to the structured view, emit a "
        "coarse context_adjustment (negative, neutral, or positive), a context_materiality (0-3), "
        "a veto flag (true only if the context makes the trade unattractive regardless of score), "
        "and a reason_code. Default to neutral when the context does not change your view."
    )


def adjustment_value(context_adjustment: str) -> int:
    """Map a coarse context_adjustment to its numeric value (-1/0/+1)."""
    try:
        return ADJUSTMENT_VALUE[context_adjustment]
    except KeyError:
        raise ValueError(
            f"context_adjustment must be one of {COARSE_ADJUSTMENTS}, got {context_adjustment!r}"
        ) from None


def final_conviction(structured_conviction: int, context_adjustment: str) -> int:
    """Mechanical sum: structured_conviction + numeric(adjustment), clamped to [1, 10]."""
    raw = structured_conviction + adjustment_value(context_adjustment)
    return max(_CONVICTION_MIN, min(_CONVICTION_MAX, raw))


# Structured tool-call schemas (typed output at the API boundary, mirroring analysis/ai_analyst).
ARM2_TOOL: dict = {
    "name": "submit_structured_score",
    "description": "Score a candidate using structured evidence only.",
    "input_schema": {
        "type": "object",
        "properties": {
            "structured_conviction": {"type": "integer", "minimum": 1, "maximum": 10},
            "structured_rank": {"type": "integer", "minimum": 1},
            "structured_reasoning": {"type": "string"},
        },
        "required": ["structured_conviction", "structured_reasoning"],
    },
}

ARM3_TOOL: dict = {
    "name": "submit_context_revision",
    "description": "Revise the structured score given timestamp-safe context.",
    "input_schema": {
        "type": "object",
        "properties": {
            "context_adjustment": {"type": "string", "enum": list(COARSE_ADJUSTMENTS)},
            "context_materiality": {"type": "integer", "minimum": 0, "maximum": 3},
            "veto": {"type": "boolean"},
            "reason_code": {"type": "string", "enum": list(REASON_CODES)},
        },
        "required": ["context_adjustment", "context_materiality", "veto", "reason_code"],
    },
}
