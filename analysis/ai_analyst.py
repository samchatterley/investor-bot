from __future__ import annotations
import json
import logging
from typing import Optional
import anthropic
from config import ANTHROPIC_API_KEY, MIN_CONFIDENCE, MAX_POSITIONS, MAX_HOLD_DAYS
from analysis.performance import get_actionable_feedback

logger = logging.getLogger(__name__)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


SYSTEM_PROMPT = """You are a quantitative short-term equity trader. Your goal is to identify
stocks likely to gain 5-15% over the next 1-5 trading days using technical analysis signals.

You focus on:
- Momentum: strong recent performance with volume confirmation
- Mean reversion: oversold conditions + Bollinger Band low + catalyst volume spike
- Trend confirmation: moving average crossovers, momentum signals aligning

You are disciplined: you only recommend trades with high conviction. You do NOT chase
already-extended moves. You protect capital — recommending SELL on positions showing
weakness is as important as finding new BUYs.

IMPORTANT: Write all reasoning in plain English that a non-technical reader can understand.
Do NOT use acronyms or jargon. Instead of "RSI at 49", say "the momentum indicator shows
the stock is neither overbought nor oversold". Instead of "MACD crossed up", say "momentum
just shifted upward". Instead of "EMA9 above EMA21", say "the short-term trend is above the
medium-term trend, confirming an uptrend". Keep reasoning to 2-3 sentences maximum.

Always respond with valid JSON only. No prose outside the JSON."""


_REGIME_ADVICE = {
    "BULL_TRENDING": "Market is trending upward — favour momentum and trend-continuation setups.",
    "CHOPPY":        "Market is choppy with no clear direction — favour mean-reversion (oversold bounces). Avoid chasing moves.",
    "HIGH_VOL":      "High volatility with a weakening market — only the highest-conviction setups. Be conservative with confidence scores.",
    "BEAR_DAY":      "BEAR DAY — SPY down sharply. NO new BUYs.",
    "UNKNOWN":       "",
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
    lessons: list[str] | None = None,
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
        macro_block = f"⚠️  MACRO EVENT TODAY: {macro_risk['event']} — avoid new positions, vol is elevated.\n"

    # VIX
    vix_block = ""
    if vix is not None:
        tone = "HIGH — widen stops, reduce size" if vix > 25 else "ELEVATED" if vix > 18 else "LOW — normal conditions"
        vix_block = f"VIX: {vix:.1f} ({tone})\n"

    # Earnings risk
    earnings_block = ""
    if earnings_risk:
        lines = [f"  ⚠️  {sym}: earnings {str(ed)}" for sym, ed in earnings_risk.items()]
        earnings_block = "EARNINGS RISK (EXIT THESE POSITIONS — do not hold through earnings):\n" + "\n".join(lines) + "\n"

    # Sector performance
    sector_block = ""
    if sector_performance:
        top = list(sector_performance.items())[:3]
        bot = list(sector_performance.items())[-3:]
        sector_block = (
            "SECTOR ROTATION (5-day performance):\n"
            + "  Leading: " + ", ".join(f"{s} {r:+.1f}%" for s, r in top) + "\n"
            + "  Lagging: " + ", ".join(f"{s} {r:+.1f}%" for s, r in reversed(bot)) + "\n"
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

    # Weekly review lessons
    lessons_block = ""
    if lessons:
        lessons_block = "LESSONS FROM LAST WEEK'S REVIEW (apply these adjustments today):\n"
        lessons_block += "\n".join(f"  - {l}" for l in lessons) + "\n"

    # Stale positions note
    stale_block = ""
    if stale_positions:
        stale_block = f"""
STALE POSITIONS (held ≥ {MAX_HOLD_DAYS} trading days — consider exiting to free capital):
{', '.join(stale_positions)}
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
            options_block = "OPTIONS FLOW (large call buying vs put buying signals where informed money is positioned):\n" + "\n".join(lines) + "\n"

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
            pnl = f"{'+'if r['daily_pnl_usd']>=0 else ''}${r['daily_pnl_usd']:.2f}"
            trade_strs = [f"{t['symbol']} {t['action']}" for t in r.get("trades", [])]
            trades_str = ", ".join(trade_strs) if trade_strs else "no trades"
            lines.append(f"  {r['date']} P&L {pnl}: {trades_str}")
        track_block = "YOUR RECENT TRADING HISTORY (learn from this):\n" + "\n".join(lines) + "\n"

    # Position ages
    ages_block = ""
    if position_ages:
        lines = [f"  {sym}: {age} trading day{'s' if age!=1 else ''}" for sym, age in position_ages.items()]
        if lines:
            ages_block = "CURRENT POSITION AGES:\n" + "\n".join(lines) + "\n"

    snapshots_json = json.dumps(snapshots, indent=2)

    return f"""Analyse today's market data and make trading decisions.

{regime_block}{macro_block}{vix_block}{earnings_block}{sector_block}{sentiment_block}{winrate_block}{lessons_block}
PORTFOLIO STATUS:
- Available cash: ${available_cash:.2f}
- Total portfolio value: ${portfolio_value:.2f}
- Max open positions allowed: {MAX_POSITIONS}
- Current open positions: {json.dumps(current_positions, indent=2)}

{ages_block}
{stale_block}
{track_block}
{options_block}
{news_block}
TODAY'S MARKET SNAPSHOTS (technical indicators):
{snapshots_json}

INDICATOR GUIDE:
- rsi_14: <30 oversold (reversal up), >70 overbought (caution)
- macd_diff: positive = bullish momentum; macd_crossed_up = fresh buy signal
- ema9_above_ema21: true = uptrend confirmed
- bb_pct: 0.0 = at lower band (oversold), 1.0 = at upper band (overbought)
- vol_ratio: >1.5 = elevated volume confirms the move, <0.7 = low conviction
- price_vs_ema9_pct: distance from 9-day EMA
- weekly_trend_up: true = weekly trend is upward (buy candidates failing this are blocked upstream)

PRE-FILTER NOTE: All buy candidates in the snapshots below have already passed a rule-based
technical screen (confirmed momentum or oversold signal with volume). Your role is to rank
them by conviction, incorporate news and options context, and identify the 1-{MAX_POSITIONS}
highest quality setups. Current positions are listed separately for HOLD/SELL decisions.

TASK:
1. For each CURRENT POSITION decide HOLD or SELL — factor in age, news, and momentum
2. From the buy candidates, select up to {MAX_POSITIONS} (skip all if bear market filter active)
3. Only recommend BUY if confidence >= {MIN_CONFIDENCE}/10
4. Treat unusual options call activity as a supporting signal, not a standalone reason to buy

NOTE: Do NOT include a cash_fraction field — position sizing is handled automatically
by the Kelly Criterion using your confidence score. Focus on signal quality and confidence accuracy.

Respond with ONLY this JSON:
{{
  "date": "YYYY-MM-DD",
  "market_summary": "one sentence on overall market tone",
  "position_decisions": [
    {{
      "symbol": "AAPL",
      "action": "HOLD" | "SELL",
      "confidence": 8,
      "reasoning": "brief reason"
    }}
  ],
  "buy_candidates": [
    {{
      "symbol": "MSFT",
      "confidence": 8,
      "reasoning": "brief reason",
      "key_signal": "rsi_oversold | macd_crossover | momentum | mean_reversion | trend_continuation | news_catalyst"
    }}
  ]
}}"""


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
    lessons: list[str] | None = None,
) -> Optional[dict]:
    prompt = build_prompt(
        snapshots, current_positions, available_cash, portfolio_value,
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
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        decisions = json.loads(raw)
        logger.info(f"AI analysis complete. Market: {decisions.get('market_summary', '')}")
        return decisions

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse AI response as JSON: {e}")
        logger.debug(f"Raw response: {raw if 'raw' in dir() else 'unknown'}")
        return None
    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in AI analysis: {e}")
        return None
