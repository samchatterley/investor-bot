"""Context card — distill a candidate's signals into a compact, prompt-affordable summary.

The daily decision prompt is already ~379k tokens, so we can't feed raw filings/news/transcripts. A
context card is the distillation layer: per candidate, a few short lines assembled from
already-distilled signals under a hard character budget. It is the single plug point for richer text
contributors (8-K / news / transcript summaries, Lazy Prices filing change) and the input that lets
the LLM's *synthesis* weigh signals together rather than reading scattered boolean fields.

Design: each contributor is `(symbol, snapshot, context) -> str | None`; the assembler joins the
non-empty lines under `max_chars`. Contributors are fail-safe (an exception yields no line, never
breaks the card). NOT yet wired into the live prompt — that's a freeze-gated experiment arm.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

Contributor = Callable[[str, dict, dict | None], "str | None"]

# Snapshot flag -> human label (mirrors the catalyst signals; kept terse for the token budget).
_BEARISH_CATALYSTS: dict[str, str] = {
    "accounting_concern": "accounting concern",
    "guidance_negative": "guidance cut",
    "secondary_offering": "secondary offering",
    "insider_sell_cluster": "insider selling",
    "analyst_downgrade": "analyst downgrade",
    "eps_estimate_cut": "EPS estimate cut",
    "index_deletion": "index deletion",
}
_BULLISH_CATALYSTS: dict[str, str] = {
    "insider_cluster": "insider buying",
    "guidance_raise": "guidance raise",
    "analyst_upgrade": "analyst upgrade",
    "activist_filing": "activist 13D",
}

# change_score at/above which the Lazy Prices line is worth surfacing (provisional — calibrate from
# the backtest; live large-caps sit at 0.005-0.035, so 0.10 flags a genuinely large filing change).
_FILING_CHANGE_MIN = 0.10


def _catalyst_line(symbol: str, snapshot: dict, context: dict | None) -> str | None:
    bear = [label for flag, label in _BEARISH_CATALYSTS.items() if snapshot.get(flag)]
    bull = [label for flag, label in _BULLISH_CATALYSTS.items() if snapshot.get(flag)]
    parts = []
    if bear:
        parts.append("bearish: " + ", ".join(bear))
    if bull:
        parts.append("bullish: " + ", ".join(bull))
    return " | ".join(parts) or None


def _short_interest_line(symbol: str, snapshot: dict, context: dict | None) -> str | None:
    spf = snapshot.get("short_pct_float")
    if not spf:
        return None
    line = f"short interest {spf * 100:.0f}% of float"
    dtc = snapshot.get("days_to_cover")
    if dtc:
        line += f", {dtc:.1f}d to cover"
    return line


def _filing_change_line(symbol: str, snapshot: dict, context: dict | None) -> str | None:
    rec = (context or {}).get("filing_change")
    if not rec or rec.get("change_score", 0.0) < _FILING_CHANGE_MIN:
        return None
    return f"10-K language changed {rec['change_score'] * 100:.0f}% YoY (Lazy Prices: elevated)"


DEFAULT_CONTRIBUTORS: tuple[Contributor, ...] = (
    _catalyst_line,
    _short_interest_line,
    _filing_change_line,
)


def build_context_card(
    symbol: str,
    snapshot: dict,
    *,
    contributors: Iterable[Contributor] = DEFAULT_CONTRIBUTORS,
    context: dict | None = None,
    max_chars: int = 400,
) -> str:
    """Assemble a compact context card for `symbol`. Empty string when nothing material to say."""
    lines: list[str] = []
    for contributor in contributors:
        try:
            line = contributor(symbol, snapshot, context)
        except Exception:
            line = None  # fail-safe: a broken contributor must never break the card
        if line:
            lines.append(f"• {line}")
    if not lines:
        return ""
    return f"{symbol} context:\n" + "\n".join(lines)[: max_chars - len(symbol) - 10]
