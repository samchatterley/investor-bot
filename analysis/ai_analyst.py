import hashlib
import json
import logging
from collections.abc import Sequence
from datetime import UTC

import anthropic

from analysis.performance import get_actionable_feedback
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, MAX_HOLD_DAYS, MAX_POSITIONS, MIN_CONFIDENCE
from utils.validators import validate_ai_response

logger = logging.getLogger(__name__)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


SYSTEM_PROMPT = """You are a short-term US equities decision-support analyst.
Your job is to rank pre-filtered candidates by signal quality, downside risk, and uncertainty.
Default to no new BUY when evidence is mixed, stale, contradictory, or mostly broad-market beta.

You focus on these signal families:
- Momentum: strong recent performance with volume confirmation
- Mean reversion: oversold conditions + Bollinger Band low + catalyst volume spike
- Trend confirmation: moving average crossovers, momentum signals aligning
- Volatility squeeze (bb_squeeze): Bollinger Bands contract to a multi-week low, then expand
  with directional confirmation — enter on the breakout
- 52-week high breakout (breakout_52w): price within 3% of yearly high, above-average volume,
  weekly trend intact — growth/momentum continuation
- Relative strength leader (rs_leader): stock consistently outperforming SPY over both 5 and 10
  days with EMA alignment — buy the market leader, not the laggard
- Inside-day breakout (inside_day_breakout): the prior bar's full range contained today's; when
  price breaks out of that compression with volume, it often accelerates
- Trend pullback (trend_pullback): uptrend intact (EMA9 > EMA21), price has pulled back to
  within 0.5–3% below EMA21, RSI between 40–58 — buy the dip in a healthy trend
- VWAP reclaim (vwap_reclaim): price moved above VWAP intraday with positive momentum and not
  extended — institutional support confirmed, high-probability continuation
- Opening range breakout (orb_breakout): price broke above the first-30-minute high with
  above-average volume — classic intraday momentum signal valid all session
- Intraday momentum (intraday_momentum): stock up >2% from open, above VWAP, intraday RSI not
  overbought, and daily trend confirms — catches moves that develop during the session

You are disciplined: you only recommend trades with high conviction. You do NOT chase
already-extended moves. You protect capital — recommending SELL on positions showing
weakness is as important as finding new BUYs.

Evidence weighting:
1. Price/volume confirmation beats headlines, options flow, and sentiment.
2. Options and sentiment are supporting evidence only — never a standalone reason to buy.
3. Stale, contradictory, or broad-market-dependent evidence lowers confidence.

Confidence calibration (use evidence quality, not excitement):
- 10: rare — multiple independent confirmations, clean regime, no data gaps
- 9: strong setup with minor uncertainty
- 8: good setup, clear invalidation trigger, acceptable risk
- 7: marginal — only if no major flags present
- 6 or below: do not recommend as BUY

For each decision provide TWO fields:
- reasoning: precise technical rationale. Use exact indicator values, levels, and signal
  confluence (e.g. "RSI at 31 with MACD diff crossing positive on 1.8x average volume;
  EMA9 recrossed EMA21 — classic mean-reversion setup with volume confirmation"). Be specific.
- summary: one plain-English sentence (no jargon) suitable for a non-technical reader."""


# Tool schema — forces Claude to return structured data matching this shape.
# Eliminates JSON parsing fragility; confidence bounds and signal enum are
# enforced at the API layer before the response reaches our validator.
_DECISION_TOOL = {
    "name": "submit_trading_decisions",
    "description": "Submit structured trading decisions for the current session",
    "input_schema": {
        "type": "object",
        "properties": {
            "date": {"type": "string", "description": "Today's date YYYY-MM-DD"},
            "market_summary": {
                "type": "string",
                "description": "One sentence on overall market tone",
            },
            "position_decisions": {
                "type": "array",
                "description": "Decision for each currently held position",
                "items": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "action": {"type": "string", "enum": ["HOLD", "SELL"]},
                        "confidence": {"type": "integer", "minimum": 1, "maximum": 10},
                        "reasoning": {
                            "type": "string",
                            "description": "Precise technical rationale with indicator values",
                        },
                        "summary": {
                            "type": "string",
                            "description": "One plain-English sentence for non-technical readers",
                        },
                    },
                    "required": ["symbol", "action", "confidence", "reasoning", "summary"],
                },
            },
            "buy_candidates": {
                "type": "array",
                "description": "Ranked list of buy candidates, highest conviction first. Must NOT contain any symbol already in current open positions.",
                "items": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "confidence": {"type": "integer", "minimum": 1, "maximum": 10},
                        "reasoning": {
                            "type": "string",
                            "description": "Precise technical rationale with indicator values",
                        },
                        "summary": {
                            "type": "string",
                            "description": "One plain-English sentence for non-technical readers",
                        },
                        "key_signal": {
                            "type": "string",
                            "enum": [
                                "mean_reversion",
                                "momentum",
                                "trend_continuation",
                                "macd_crossover",
                                "rsi_oversold",
                                "news_catalyst",
                                "bb_squeeze",
                                "breakout_52w",
                                "rs_leader",
                                "inside_day_breakout",
                                "trend_pullback",
                                "vwap_reclaim",
                                "orb_breakout",
                                "intraday_momentum",
                                "unknown",
                            ],
                        },
                        "do_nothing_case": {
                            "type": "string",
                            "description": "Strongest reason not to take this trade",
                        },
                        "invalidation_trigger": {
                            "type": "string",
                            "description": "Specific condition that would invalidate this recommendation",
                        },
                    },
                    "required": [
                        "symbol",
                        "confidence",
                        "reasoning",
                        "summary",
                        "key_signal",
                        "do_nothing_case",
                        "invalidation_trigger",
                    ],
                },
            },
        },
        "required": ["date", "market_summary", "position_decisions", "buy_candidates"],
    },
}


_REGIME_ADVICE = {
    "BULL_TRENDING": "Market is trending upward — favour momentum and trend-continuation setups.",
    "CHOPPY": "Market is choppy with no clear direction — favour mean-reversion (oversold bounces). Avoid chasing moves.",
    "HIGH_VOL": "High volatility with a weakening market — only the highest-conviction setups. Be conservative with confidence scores.",
    "BEAR_DAY": "BEAR DAY — SPY down sharply. NO new BUYs.",
    "UNKNOWN": "",
}


def build_prompt(
    snapshots: list[dict],
    current_positions: list[dict],
    available_cash: float,
    portfolio_value: float,
    news_by_symbol: dict | None = None,
    track_record: list[dict] | None = None,
    market_regime: dict | None = None,
    position_ages: dict | None = None,
    stale_positions: list[str] | None = None,
    vix: float | None = None,
    sector_performance: dict | None = None,
    sentiment: dict | None = None,
    earnings_risk: dict | None = None,
    macro_risk: dict | None = None,
    leading_sectors: list[str] | None = None,
    options_signals: dict | None = None,
    lessons: Sequence[str | dict] | None = None,
) -> str:

    # Market regime (4-state)
    regime_block = ""
    if market_regime:
        regime_name = market_regime.get("regime", "UNKNOWN")
        spy_1d = market_regime.get("spy_change_pct", 0.0)
        spy_5d = market_regime.get("spy_5d_pct")
        advice = _REGIME_ADVICE.get(regime_name, "")
        if market_regime.get("is_bearish"):
            regime_block = f"⚠️  BEAR DAY: SPY {spy_1d:+.1f}% today. NO new BUYs.\n"
        else:
            week_str = f", {spy_5d:+.1f}% this week" if spy_5d is not None else ""
            regime_block = f"MARKET REGIME: {regime_name} (SPY {spy_1d:+.1f}% today{week_str})\n"
            if advice:
                regime_block += f"→ {advice}\n"

    # Macro risk
    macro_block = ""
    if macro_risk and macro_risk.get("is_high_risk"):
        macro_block = (
            f"⚠️  MACRO EVENT TODAY: {macro_risk['event']} — avoid new positions, vol is elevated.\n"
        )

    # VIX
    vix_block = ""
    if vix is not None:
        tone = (
            "HIGH — widen stops, reduce size"
            if vix > 25
            else "ELEVATED"
            if vix > 18
            else "LOW — normal conditions"
        )
        vix_block = f"VIX: {vix:.1f} ({tone})\n"

    # Earnings risk
    earnings_block = ""
    if earnings_risk:
        lines = [f"  ⚠️  {sym}: earnings {str(ed)}" for sym, ed in earnings_risk.items()]
        earnings_block = (
            "EARNINGS RISK (EXIT THESE POSITIONS — do not hold through earnings):\n"
            + "\n".join(lines)
            + "\n"
        )

    # Sector performance
    sector_block = ""
    if sector_performance:
        top = list(sector_performance.items())[:3]
        bot = list(sector_performance.items())[-3:]
        sector_block = (
            "SECTOR ROTATION (5-day performance):\n"
            + "  Leading: "
            + ", ".join(f"{s} {r:+.1f}%" for s, r in top)
            + "\n"
            + "  Lagging: "
            + ", ".join(f"{s} {r:+.1f}%" for s, r in reversed(bot))
            + "\n"
        )
        if leading_sectors:
            sector_block += f"  Favour candidates from: {', '.join(leading_sectors)}\n"

    # Social sentiment
    sentiment_block = ""
    if sentiment:
        lines = []
        for sym, data in sentiment.items():
            bull = data.get("bullish_pct", 0)
            bear = data.get("bearish_pct", 0)
            tone = "bullish" if bull > 60 else "bearish" if bear > 60 else "neutral"
            lines.append(f"  {sym}: {bull}% bullish / {bear}% bearish ({tone})")
        if lines:
            sentiment_block = "SOCIAL SENTIMENT:\n" + "\n".join(lines) + "\n"

    # Performance feedback (regime-aware, actionable directives)
    winrate_block = get_actionable_feedback()

    # Weekly review lessons — support both plain strings (legacy) and structured dicts
    lessons_block = ""
    if lessons:
        lessons_block = "LESSONS FROM LAST WEEK'S REVIEW (apply these adjustments today):\n"
        for item in lessons:
            text = item["lesson"] if isinstance(item, dict) else item
            lessons_block += f"  - {text}\n"

    # Stale positions note
    stale_block = ""
    if stale_positions:
        stale_block = f"""
STALE POSITIONS (held ≥ {MAX_HOLD_DAYS} trading days — consider exiting to free capital):
{", ".join(stale_positions)}
"""

    # Options flow
    options_block = ""
    if options_signals:
        lines = []
        for sym, data in options_signals.items():
            pc = data.get("put_call_ratio", 1.0)
            unusual = data.get("unusual_calls", False)
            tone = "bullish" if pc < 0.7 else "bearish" if pc > 1.3 else "neutral"
            flag = "  ⚡ UNUSUAL CALL ACTIVITY" if unusual else ""
            lines.append(f"  {sym}: put/call ratio {pc:.2f} ({tone}){flag}")
        if lines:
            options_block = (
                "OPTIONS FLOW (large call buying vs put buying signals where informed money is positioned):\n"
                + "\n".join(lines)
                + "\n"
            )

    # News section
    news_block = ""
    if news_by_symbol:
        lines = []
        for sym, headlines in news_by_symbol.items():
            if headlines:
                lines.append(f"  {sym}: " + " | ".join(headlines))
        if lines:
            news_block = "RECENT NEWS HEADLINES:\n" + "\n".join(lines) + "\n"

    # Track record section
    track_block = ""
    if track_record:
        lines = []
        for r in track_record:
            pnl = f"{'+' if r['daily_pnl_usd'] >= 0 else ''}${r['daily_pnl_usd']:.2f}"
            trade_strs = [f"{t['symbol']} {t['action']}" for t in r.get("trades", [])]
            trades_str = ", ".join(trade_strs) if trade_strs else "no trades"
            lines.append(f"  {r['date']} P&L {pnl}: {trades_str}")
        track_block = "YOUR RECENT TRADING HISTORY (learn from this):\n" + "\n".join(lines) + "\n"

    # Position ages
    ages_block = ""
    if position_ages:
        lines = [
            f"  {sym}: {age} trading day{'s' if age != 1 else ''}"
            for sym, age in position_ages.items()
        ]
        if lines:
            ages_block = "CURRENT POSITION AGES:\n" + "\n".join(lines) + "\n"

    held_symbols = sorted({p["symbol"] for p in current_positions}) if current_positions else []
    held_block = (
        f"ALREADY HELD — DO NOT include in buy_candidates: {', '.join(held_symbols)}\n"
        if held_symbols
        else ""
    )

    snapshots_json = json.dumps(snapshots, indent=2)

    return f"""Analyse today's market data and make trading decisions.

{regime_block}{macro_block}{vix_block}{earnings_block}{sector_block}{sentiment_block}{winrate_block}{lessons_block}
PORTFOLIO STATUS:
- Available cash: ${available_cash:.2f}
- Total portfolio value: ${portfolio_value:.2f}
- Max open positions allowed: {MAX_POSITIONS}
- Current open positions: {json.dumps(current_positions, indent=2)}

{held_block}{ages_block}
{stale_block}
{track_block}
{options_block}
{news_block}
TODAY'S MARKET SNAPSHOTS (technical indicators):
{snapshots_json}

INDICATOR GUIDE:
Daily indicators (based on close prices):
- rsi_14: <30 oversold (reversal up), >70 overbought (caution)
- macd_diff: positive = bullish momentum; macd_crossed_up = fresh buy signal
- ema9_above_ema21: true = uptrend confirmed
- bb_pct: 0.0 = at lower band (oversold), 1.0 = at upper band (overbought)
- bb_squeeze: true = Bollinger Bands compressed to multi-week low — volatility expansion imminent
- vol_ratio: >1.5 = elevated volume confirms the move, <0.7 = low conviction
- price_vs_ema9_pct: distance from 9-day EMA
- price_vs_ema21_pct: distance from 21-day EMA (negative = below EMA21; -0.5 to -3% = pullback zone)
- price_vs_52w_high_pct: distance from 52-week high (0 = at high; -3% to 0 = near-high breakout zone)
- rel_strength_5d: 5-day return minus SPY return (positive = outperforming market)
- rel_strength_10d: 10-day return minus SPY return (positive = sustained outperformance)
- is_inside_day: true = today's range was contained within the previous day's range (coiling)
- weekly_trend_up: true = weekly trend is upward (buy candidates failing this are blocked upstream)

Intraday indicators (from Alpaca minute bars; absent if market not yet open or data unavailable):
- gap_pct: (today's open − yesterday's close) / yesterday's close × 100; positive = gap up
- intraday_change_pct: (current price − today's open) / open × 100; measures post-open momentum
- price_above_vwap / pct_vs_vwap: position relative to VWAP; above = institutional bid side
- orb_high / orb_low: first-30-minute range high and low
- orb_breakout_up: true = price broke above orb_high with volume confirmation
- intraday_rsi: RSI-14 on 5-minute bars; >75 = short-term overbought, <30 = short-term oversold
- intraday_cumvol: cumulative shares traded today (compare to avg_volume for pacing context)

PRE-FILTER NOTE: All buy candidates in the snapshots below have already passed a rule-based
technical screen (confirmed momentum or oversold signal with volume). Your role is to rank
them by conviction, incorporate news and options context, and identify the 1-{MAX_POSITIONS}
highest quality setups. Current positions are listed separately for HOLD/SELL decisions.

TASK:
1. For each CURRENT POSITION decide HOLD or SELL — factor in age, news, and momentum
2. From the buy candidates, select up to {MAX_POSITIONS}{" — never include a symbol from ALREADY HELD above" if held_symbols else ""}
3. Only recommend BUY if confidence >= {MIN_CONFIDENCE}/10
4. Treat unusual options call activity as a supporting signal, not a standalone reason to buy
5. For every BUY candidate, populate do_nothing_case honestly — if you cannot clearly refute it, omit the candidate

NOTE: Do NOT include a cash_fraction field — position sizing is handled automatically.
Focus on signal quality and confidence accuracy.

Use the submit_trading_decisions tool to return your analysis."""


_COST_PER_M_INPUT = 3.0  # USD per 1M input tokens (claude-sonnet-4-x)
_COST_PER_M_OUTPUT = 15.0  # USD per 1M output tokens


def _record_llm_usage(run_id: str | None, input_tokens: int, output_tokens: int):
    """Write token counts and estimated cost to the llm_usage SQLite table."""
    cost = (input_tokens / 1_000_000) * _COST_PER_M_INPUT + (
        output_tokens / 1_000_000
    ) * _COST_PER_M_OUTPUT
    try:
        from datetime import datetime

        from utils.db import get_db

        ts = datetime.now(UTC).isoformat()
        with get_db() as conn:
            conn.execute(
                "INSERT INTO llm_usage (run_id, ts, model, input_tokens, output_tokens, cost_usd) "
                "VALUES (?,?,?,?,?,?)",
                (run_id, ts, CLAUDE_MODEL, input_tokens, output_tokens, round(cost, 6)),
            )
        logger.info(
            f"LLM usage: {input_tokens} in / {output_tokens} out  ≈ ${cost:.4f}  run_id={run_id}"
        )
    except Exception as e:
        logger.warning(f"Failed to record LLM usage: {e}")


def _record_llm_call_audit(run_id: str | None, prompt_hash: str, raw_response: str) -> None:
    """Append a compact LLM call record to the audit trail.

    Stores prompt_hash (first 16 hex chars of SHA-256) and the raw tool-call
    response so every trade is explainable: which prompt produced which response,
    which validator accepted it, and which order resulted.
    """
    try:
        from utils.audit_log import log_event

        log_event(
            "LLM_CALL",
            {
                "run_id": run_id,
                "model": CLAUDE_MODEL,
                "prompt_hash": prompt_hash,
                "raw_response_snippet": raw_response[:500],  # first 500 chars for quick inspection
            },
        )
    except Exception as e:
        logger.warning(f"_record_llm_call_audit failed: {e}")


def get_trading_decisions(
    snapshots: list[dict],
    current_positions: list[dict],
    available_cash: float,
    portfolio_value: float,
    news_by_symbol: dict | None = None,
    track_record: list[dict] | None = None,
    market_regime: dict | None = None,
    position_ages: dict | None = None,
    stale_positions: list[str] | None = None,
    vix: float | None = None,
    sector_performance: dict | None = None,
    sentiment: dict | None = None,
    earnings_risk: dict | None = None,
    macro_risk: dict | None = None,
    leading_sectors: list[str] | None = None,
    options_signals: dict | None = None,
    lessons: Sequence[str | dict] | None = None,
    run_id: str | None = None,
) -> dict | None:
    prompt = build_prompt(
        snapshots,
        current_positions,
        available_cash,
        portfolio_value,
        news_by_symbol=news_by_symbol,
        track_record=track_record,
        market_regime=market_regime,
        position_ages=position_ages,
        stale_positions=stale_positions,
        vix=vix,
        sector_performance=sector_performance,
        sentiment=sentiment,
        earnings_risk=earnings_risk,
        macro_risk=macro_risk,
        leading_sectors=leading_sectors,
        options_signals=options_signals,
        lessons=lessons,
    )

    try:
        response = client.messages.create(  # type: ignore[call-overload]
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[_DECISION_TOOL],
            tool_choice={"type": "tool", "name": "submit_trading_decisions"},
            messages=[{"role": "user", "content": prompt}],
        )

        # Record token usage immediately — even if subsequent parsing fails
        if hasattr(response, "usage") and response.usage:
            _record_llm_usage(run_id, response.usage.input_tokens, response.usage.output_tokens)

        # Audit trail — hash the prompt and record raw response for post-hoc explainability.
        # Every trade can be traced: "this order happened because Claude returned X,
        # validator accepted it, and risk gate approved it."
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        raw_content = json.dumps(
            [b.input if hasattr(b, "input") else str(b) for b in response.content]
        )
        _record_llm_call_audit(run_id, prompt_hash, raw_content)

        tool_block = next((b for b in response.content if hasattr(b, "input")), None)
        if tool_block is None:
            logger.error("AI response contained no tool call")
            return None

        decisions = tool_block.input

        # Independent domain validation — API schema enforcement and internal
        # validation are kept separate so neither can mask failures in the other.
        # known_symbols is derived from the snapshots passed to this call.
        known_symbols = {s["symbol"] for s in snapshots} | {p["symbol"] for p in current_positions}
        is_valid, errors = validate_ai_response(decisions, known_symbols)
        if not is_valid:
            logger.warning(f"AI response failed domain validation: {errors}")
            # Return decisions anyway — main.py is the authoritative gate and will
            # fail closed (block all Claude-driven orders) on any validation errors.

        logger.info(f"AI analysis complete. Market: {decisions.get('market_summary', '')}")
        return decisions

    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in AI analysis: {e}")
        return None
