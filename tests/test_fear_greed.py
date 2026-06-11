"""Tests for data/fear_greed.py."""

import pytest

from data.fear_greed import (
    _aaii_component,
    _breadth_component,
    _momentum_component,
    _nhl_component,
    _vix_component,
    compute_fear_greed_index,
    is_excessive_greed,
    is_extreme_fear,
)


class TestVixComponent:
    def test_low_vix_gives_high_score(self):
        # VIX 10 → score 90 (greed)
        assert _vix_component(10.0) == pytest.approx(90.0)

    def test_high_vix_gives_low_score(self):
        # VIX 40 → score 10 (fear)
        assert _vix_component(40.0) == pytest.approx(10.0)

    def test_mid_vix_interpolates(self):
        # VIX 25 → midpoint
        score = _vix_component(25.0)
        assert 0.0 < score < 100.0

    def test_clamps_below_zero(self):
        # Extreme VIX should never go below 0
        assert _vix_component(100.0) == 0.0

    def test_clamps_above_100(self):
        # Very low VIX should never exceed 100
        assert _vix_component(1.0) == 100.0

    def test_none_returns_neutral(self):
        assert _vix_component(None) == 50.0


class TestAaiiComponent:
    def test_low_bears_gives_high_score(self):
        # Bears 20% → score 90
        assert _aaii_component(20.0) == pytest.approx(90.0)

    def test_high_bears_gives_low_score(self):
        # Bears 55% → score 10
        assert _aaii_component(55.0) == pytest.approx(10.0)

    def test_clamps_below_zero(self):
        assert _aaii_component(80.0) == 0.0

    def test_clamps_above_100(self):
        assert _aaii_component(0.0) == 100.0

    def test_none_returns_neutral(self):
        assert _aaii_component(None) == 50.0

    def test_moderate_bears_mid_range(self):
        score = _aaii_component(37.5)
        assert 0.0 < score < 100.0


class TestNhlComponent:
    def test_low_ratio_fear(self):
        # Ratio 0.1 → score 10
        assert _nhl_component(0.1) == pytest.approx(10.0)

    def test_high_ratio_greed(self):
        # Ratio 5.0 → score 90
        assert _nhl_component(5.0) == pytest.approx(90.0)

    def test_none_returns_neutral(self):
        assert _nhl_component(None) == 50.0

    def test_clamps_below_zero(self):
        # nhl_ratio well below 0.1 should still give a very low score (clamped at 0)
        assert _nhl_component(-10.0) == 0.0

    def test_clamps_above_100(self):
        assert _nhl_component(100.0) == 100.0


class TestMomentumComponent:
    def test_negative_return_fear(self):
        # -10% return → score 10
        assert _momentum_component(-10.0) == pytest.approx(10.0)

    def test_positive_return_greed(self):
        # +10% return → score 90
        assert _momentum_component(10.0) == pytest.approx(90.0)

    def test_flat_return_neutral(self):
        assert _momentum_component(0.0) == pytest.approx(50.0)

    def test_none_returns_neutral(self):
        assert _momentum_component(None) == 50.0

    def test_clamps_below_zero(self):
        assert _momentum_component(-100.0) == 0.0

    def test_clamps_above_100(self):
        assert _momentum_component(100.0) == 100.0


class TestBreadthComponent:
    def test_low_breadth_fear(self):
        assert _breadth_component(20.0) == pytest.approx(20.0)

    def test_high_breadth_greed(self):
        assert _breadth_component(80.0) == pytest.approx(80.0)

    def test_none_returns_neutral(self):
        assert _breadth_component(None) == 50.0

    def test_clamps_below_zero(self):
        assert _breadth_component(-10.0) == 0.0

    def test_clamps_above_100(self):
        assert _breadth_component(110.0) == 100.0


class TestComputeFearGreedIndex:
    def test_all_none_returns_50(self):
        # All components neutral → composite = 50
        score = compute_fear_greed_index()
        assert score == pytest.approx(50.0)

    def test_extreme_fear_scenario(self):
        # High VIX, high bears, low NHL, falling SPY, narrow breadth
        score = compute_fear_greed_index(
            vix=40.0,
            aaii_bears_pct=55.0,
            nhl_ratio=0.1,
            spy_ret_20d=-10.0,
            breadth_pct_above_50d=15.0,
        )
        assert score < 20.0

    def test_extreme_greed_scenario(self):
        # Low VIX, low bears, high NHL, rising SPY, broad breadth
        score = compute_fear_greed_index(
            vix=10.0,
            aaii_bears_pct=15.0,
            nhl_ratio=5.0,
            spy_ret_20d=10.0,
            breadth_pct_above_50d=85.0,
        )
        assert score > 80.0

    def test_result_in_0_100_range(self):
        score = compute_fear_greed_index(vix=20.0, aaii_bears_pct=30.0)
        assert 0.0 <= score <= 100.0

    def test_partial_inputs(self):
        # Only some inputs provided — rest default to neutral 50
        score = compute_fear_greed_index(vix=30.0)
        assert 0.0 <= score <= 100.0

    def test_vix_weight_dominates(self):
        # High VIX should drive score down even with other inputs neutral
        score_high_vix = compute_fear_greed_index(vix=40.0)
        score_low_vix = compute_fear_greed_index(vix=10.0)
        assert score_low_vix > score_high_vix


class TestIsExtremeFear:
    def test_score_below_20_is_extreme_fear(self):
        assert is_extreme_fear(15.0) is True

    def test_score_at_20_is_not_extreme_fear(self):
        assert is_extreme_fear(20.0) is False

    def test_score_above_20_is_not_extreme_fear(self):
        assert is_extreme_fear(50.0) is False

    def test_score_zero_is_extreme_fear(self):
        assert is_extreme_fear(0.0) is True


class TestIsExcessiveGreed:
    def test_score_above_80_is_excessive_greed(self):
        assert is_excessive_greed(85.0) is True

    def test_score_at_80_is_not_excessive_greed(self):
        assert is_excessive_greed(80.0) is False

    def test_score_below_80_is_not_excessive_greed(self):
        assert is_excessive_greed(50.0) is False

    def test_score_100_is_excessive_greed(self):
        assert is_excessive_greed(100.0) is True
