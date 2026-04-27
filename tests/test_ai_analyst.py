"""Tests for analysis/ai_analyst.py — build_prompt and get_trading_decisions."""
import unittest
from unittest.mock import MagicMock, patch


def _snapshot(symbol="AAPL", price=180.0):
    return {
        "symbol": symbol, "current_price": price,
        "rsi_14": 52.0, "macd_diff": 0.05, "ema9_above_ema21": True,
        "bb_pct": 0.45, "vol_ratio": 1.2, "weekly_trend_up": True,
    }


def _decisions_response(market="Quiet", buys=None, positions=None):
    return {
        "date": "2026-01-15",
        "market_summary": market,
        "buy_candidates": buys or [],
        "position_decisions": positions or [],
    }


class TestBuildPrompt(unittest.TestCase):

    def _build(self, **kwargs):
        from analysis.ai_analyst import build_prompt
        defaults = dict(
            snapshots=[_snapshot()],
            current_positions=[],
            available_cash=50_000.0,
            portfolio_value=100_000.0,
        )
        defaults.update(kwargs)
        return build_prompt(**defaults)

    def test_returns_string(self):
        result = self._build()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 100)

    def test_snapshot_symbol_appears_in_prompt(self):
        result = self._build(snapshots=[_snapshot("NVDA")])
        self.assertIn("NVDA", result)

    def test_cash_and_portfolio_appear(self):
        result = self._build(available_cash=38_500.0, portfolio_value=99_000.0)
        self.assertIn("38500", result)
        self.assertIn("99000", result)

    def test_bear_day_warning_when_is_bearish(self):
        regime = {"regime": "BEAR_DAY", "spy_change_pct": -2.5, "is_bearish": True}
        result = self._build(market_regime=regime)
        self.assertIn("BEAR DAY", result)
        self.assertIn("NO new BUYs", result)

    def test_regime_block_appears_when_not_bearish(self):
        regime = {"regime": "BULL_TRENDING", "spy_change_pct": 1.2, "is_bearish": False}
        result = self._build(market_regime=regime)
        self.assertIn("BULL_TRENDING", result)

    def test_vix_block_appears(self):
        result = self._build(vix=22.5)
        self.assertIn("22.5", result)
        self.assertIn("VIX", result)

    def test_vix_high_tone_above_25(self):
        result = self._build(vix=28.0)
        self.assertIn("HIGH", result)

    def test_earnings_block_appears(self):
        result = self._build(earnings_risk={"AAPL": "2026-01-17"})
        self.assertIn("EARNINGS RISK", result)
        self.assertIn("AAPL", result)

    def test_sentiment_block_appears(self):
        sentiment = {"AAPL": {"bullish_pct": 75, "bearish_pct": 25}}
        result = self._build(sentiment=sentiment)
        self.assertIn("SENTIMENT", result)
        self.assertIn("75%", result)

    def test_sentiment_tone_bullish_when_above_60(self):
        sentiment = {"NVDA": {"bullish_pct": 80, "bearish_pct": 20}}
        result = self._build(sentiment=sentiment)
        self.assertIn("bullish", result)

    def test_sentiment_tone_bearish_when_below_40(self):
        sentiment = {"INTC": {"bullish_pct": 30, "bearish_pct": 70}}
        result = self._build(sentiment=sentiment)
        self.assertIn("bearish", result)

    def test_news_block_appears(self):
        news = {"AAPL": ["Apple launches new iPhone", "Revenue beats estimates"]}
        result = self._build(news_by_symbol=news)
        self.assertIn("RECENT NEWS", result)
        self.assertIn("Apple launches", result)

    def test_stale_positions_block_appears(self):
        result = self._build(stale_positions=["MSFT", "NVDA"])
        self.assertIn("STALE", result)
        self.assertIn("MSFT", result)

    def test_options_block_appears(self):
        options = {"AAPL": {"put_call_ratio": 0.55, "unusual_calls": True}}
        result = self._build(options_signals=options)
        self.assertIn("OPTIONS FLOW", result)
        self.assertIn("UNUSUAL CALL", result)

    def test_options_bullish_tone_when_low_put_call(self):
        options = {"AAPL": {"put_call_ratio": 0.50, "unusual_calls": False}}
        result = self._build(options_signals=options)
        self.assertIn("bullish", result)

    def test_track_record_block_appears(self):
        track = [{"date": "2026-01-14", "daily_pnl_usd": 250.0, "trades": []}]
        result = self._build(track_record=track)
        self.assertIn("TRADING HISTORY", result)

    def test_lessons_block_appears(self):
        result = self._build(lessons=["Avoid chasing extended moves"])
        self.assertIn("LESSONS", result)
        self.assertIn("Avoid chasing", result)

    def test_macro_risk_block_appears(self):
        macro = {"is_high_risk": True, "event": "FOMC Rate Decision"}
        result = self._build(macro_risk=macro)
        self.assertIn("MACRO EVENT", result)
        self.assertIn("FOMC", result)

    def test_no_optional_blocks_when_none_passed(self):
        result = self._build()
        self.assertNotIn("EARNINGS RISK", result)
        self.assertNotIn("RECENT NEWS", result)
        self.assertNotIn("OPTIONS FLOW", result)


class TestGetTradingDecisions(unittest.TestCase):

    def _mock_response(self, decisions: dict):
        tool_block = MagicMock()
        tool_block.input = decisions
        response = MagicMock()
        response.content = [tool_block]
        return response

    def test_returns_decisions_dict_on_success(self):
        from analysis.ai_analyst import get_trading_decisions
        fake_decisions = _decisions_response(
            buys=[{"symbol": "AAPL", "confidence": 8, "reasoning": "good", "key_signal": "momentum"}],
        )
        mock_response = self._mock_response(fake_decisions)

        with patch("analysis.ai_analyst.client") as mock_client, \
             patch("analysis.ai_analyst.validate_ai_response", return_value=(True, [])):
            mock_client.messages.create.return_value = mock_response
            result = get_trading_decisions(
                snapshots=[_snapshot()],
                current_positions=[],
                available_cash=50_000,
                portfolio_value=100_000,
            )
        self.assertIsNotNone(result)
        self.assertIn("buy_candidates", result)

    def test_returns_none_when_no_tool_block(self):
        from analysis.ai_analyst import get_trading_decisions
        response = MagicMock()
        response.content = []  # no tool block

        with patch("analysis.ai_analyst.client") as mock_client:
            mock_client.messages.create.return_value = response
            result = get_trading_decisions(
                snapshots=[_snapshot()],
                current_positions=[],
                available_cash=50_000,
                portfolio_value=100_000,
            )
        self.assertIsNone(result)

    def test_returns_none_on_api_error(self):
        import anthropic
        from analysis.ai_analyst import get_trading_decisions
        with patch("analysis.ai_analyst.client") as mock_client:
            mock_client.messages.create.side_effect = anthropic.APIError(
                message="Rate limited", request=MagicMock(), body=None
            )
            result = get_trading_decisions(
                snapshots=[_snapshot()],
                current_positions=[],
                available_cash=50_000,
                portfolio_value=100_000,
            )
        self.assertIsNone(result)

    def test_calls_validate_ai_response(self):
        from analysis.ai_analyst import get_trading_decisions
        fake_decisions = _decisions_response()
        mock_response = self._mock_response(fake_decisions)

        with patch("analysis.ai_analyst.client") as mock_client, \
             patch("analysis.ai_analyst.validate_ai_response", return_value=(True, [])) as mock_validate:
            mock_client.messages.create.return_value = mock_response
            get_trading_decisions(
                snapshots=[_snapshot("AAPL")],
                current_positions=[{"symbol": "NVDA"}],
                available_cash=50_000,
                portfolio_value=100_000,
            )
        mock_validate.assert_called_once()
        # known_symbols should include both snapshot and position symbols
        known = mock_validate.call_args[0][1]
        self.assertIn("AAPL", known)
        self.assertIn("NVDA", known)

    def test_returns_decisions_even_when_validation_fails(self):
        from analysis.ai_analyst import get_trading_decisions
        fake_decisions = _decisions_response()
        mock_response = self._mock_response(fake_decisions)

        with patch("analysis.ai_analyst.client") as mock_client, \
             patch("analysis.ai_analyst.validate_ai_response", return_value=(False, ["bad symbol"])):
            mock_client.messages.create.return_value = mock_response
            result = get_trading_decisions(
                snapshots=[_snapshot()],
                current_positions=[],
                available_cash=50_000,
                portfolio_value=100_000,
            )
        # Should still return decisions — main.py runs the authoritative gate
        self.assertIsNotNone(result)
