"""
LLM eval tests — exercises the real AI governance layer against adversarial fixtures.

Run with: pytest evals/
"""

import json
import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _load(name: str) -> dict:
    with open(os.path.join(FIXTURES, name)) as f:
        return json.load(f)


# ── Prompt injection ──────────────────────────────────────────────────────────

class TestPromptInjectionScan:
    def test_malicious_headlines_dropped(self):
        from utils.validators import sanitize_headlines

        fixture = _load("prompt_injection_headlines.json")
        headlines = fixture["headlines"]
        # Wrap in symbol dict as the function expects
        news_input = {"AAPL": headlines}
        result = sanitize_headlines(news_input)

        kept = result.get("AAPL", [])
        kept_set = set(kept)

        for i in fixture["expected_kept_indices"]:
            assert headlines[i][:200] in kept_set, (
                f"Expected headline at index {i} to pass: {headlines[i]!r}"
            )

    def test_injection_headlines_not_in_output(self):
        from utils.validators import sanitize_headlines

        fixture = _load("prompt_injection_headlines.json")
        headlines = fixture["headlines"]
        news_input = {"AAPL": headlines}
        result = sanitize_headlines(news_input)

        kept = result.get("AAPL", [])
        for i in fixture["expected_dropped_indices"]:
            assert headlines[i] not in kept, (
                f"Injection headline at index {i} should have been dropped: {headlines[i]!r}"
            )

    def test_clean_news_unchanged(self):
        from utils.validators import sanitize_headlines

        clean = {"MSFT": ["Microsoft cloud revenue up 21%", "Strong earnings beat"]}
        result = sanitize_headlines(clean)
        assert result["MSFT"] == clean["MSFT"]


# ── Hallucinated tickers ──────────────────────────────────────────────────────

class TestHallucinatedTickers:
    def setup_method(self):
        self.fixture = _load("hallucinated_tickers.json")

    def test_hallucinated_symbols_rejected(self):
        from utils.validators import validate_ai_response

        universe = set(self.fixture["scanned_universe"])
        held = set(self.fixture["held_symbols"])
        _, errors = validate_ai_response(self.fixture["ai_response"], universe, held_symbols=held)

        error_text = " ".join(errors)
        for sym in self.fixture["expected_rejected"]:
            assert sym in error_text, f"Expected {sym!r} to appear in validation errors"

    def test_valid_symbols_not_rejected(self):
        from utils.validators import validate_ai_response

        universe = set(self.fixture["scanned_universe"])
        held = set(self.fixture["held_symbols"])
        _, errors = validate_ai_response(self.fixture["ai_response"], universe, held_symbols=held)

        error_text = " ".join(errors)
        for sym in self.fixture["expected_accepted"]:
            assert sym not in error_text, (
                f"Valid symbol {sym!r} should not appear in validation errors"
            )

    def test_partial_failure_does_not_abort_all(self):
        """Validation errors are per-candidate — valid candidates are not blocked."""
        from utils.validators import validate_ai_response

        universe = set(self.fixture["scanned_universe"])
        held = set(self.fixture["held_symbols"])
        is_valid, errors = validate_ai_response(self.fixture["ai_response"], universe, held_symbols=held)

        # is_valid is False because there were errors, but valid candidates still pass
        # (main.py only blocks buy_candidates; it does not abort the entire run on hallucination errors)
        assert not is_valid
        assert len(errors) == len(self.fixture["expected_rejected"]), (
            f"Expected exactly {len(self.fixture['expected_rejected'])} error(s)"
        )


# ── Bear market suppression ───────────────────────────────────────────────────

class TestBearMarketBuySuppression:
    def test_is_bearish_flag_set(self):
        fixture = _load("bear_market_no_buy.json")
        assert fixture["regime"]["is_bearish"] is True

    def test_buy_candidates_exist_but_are_suppressed(self):
        """The fixture has valid buy candidates, but the bear filter should block them."""
        fixture = _load("bear_market_no_buy.json")
        regime = fixture["regime"]
        ai = fixture["ai_response"]

        # Simulate the skip_buys logic from main.py
        skip_buys = regime.get("is_bearish", False)
        orders_placed = 0 if skip_buys else len(ai["buy_candidates"])

        assert skip_buys is True
        assert orders_placed == fixture["expected_buys_placed"]

    def test_bear_regime_classification(self):
        fixture = _load("bear_market_no_buy.json")
        assert fixture["regime"]["regime"] == "BEAR_DAY"
        assert fixture["regime"]["spy_change_pct"] < -1.5  # below BEAR_MARKET_SPY_THRESHOLD


# ── Conflicting signals ───────────────────────────────────────────────────────

class TestConflictingSignals:
    def setup_method(self):
        self.fixture = _load("conflicting_signals.json")

    def test_buy_while_held_rejected(self):
        from utils.validators import validate_ai_response

        universe = set(self.fixture["scanned_universe"])
        held = set(self.fixture["held_symbols"])
        _, errors = validate_ai_response(self.fixture["ai_response"], universe, held_symbols=held)

        error_text = " ".join(errors)
        for sym in self.fixture["expected_rejected_symbols"]:
            assert sym in error_text, f"Expected conflict error for held symbol {sym!r}"

    def test_non_held_symbol_still_accepted(self):
        from utils.validators import validate_ai_response

        universe = set(self.fixture["scanned_universe"])
        held = set(self.fixture["held_symbols"])
        _, errors = validate_ai_response(self.fixture["ai_response"], universe, held_symbols=held)

        error_text = " ".join(errors)
        for sym in self.fixture["expected_accepted_symbols"]:
            assert sym not in error_text, f"Non-held symbol {sym!r} should not be in errors"


# ── Malformed tool calls ──────────────────────────────────────────────────────

class TestMalformedResponses:
    @pytest.fixture(params=range(3))
    def case(self, request):
        fixture = _load("malformed_tool_calls.json")
        return fixture["cases"][request.param]

    def test_malformed_response_rejected(self, case):
        from utils.validators import validate_ai_response

        universe = {"AAPL", "MSFT", "NVDA"}
        is_valid, errors = validate_ai_response(case["response"], universe)

        if case["expected_valid"] is False:
            assert not is_valid or len(errors) > 0, (
                f"Case '{case['description']}' should have been rejected but passed"
            )


# ── Earnings risk ─────────────────────────────────────────────────────────────

class TestEarningsRisk:
    def test_fixture_structure(self):
        fixture = _load("earnings_risk.json")
        assert "expected_exits" in fixture
        assert "expected_held" in fixture
        assert len(fixture["expected_exits"]) > 0

    def test_earnings_symbols_should_exit(self):
        fixture = _load("earnings_risk.json")
        # Verify fixture consistency: every expected exit has earnings within 2 days
        for sym in fixture["expected_exits"]:
            assert sym in fixture["earnings_within_2_days"], (
                f"{sym} is in expected_exits but not in earnings_within_2_days"
            )

    def test_safe_symbols_not_exited(self):
        fixture = _load("earnings_risk.json")
        for sym in fixture["expected_held"]:
            assert sym not in fixture["earnings_within_2_days"], (
                f"{sym} is in expected_held but also in earnings_within_2_days"
            )
