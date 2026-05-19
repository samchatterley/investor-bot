"""Shared 5-state market regime classifier used by both the live pipeline
and the backtest engine.  A single implementation eliminates live/backtest
drift and provides hysteresis to prevent daily regime flip-flopping.

States (priority order):
  STRESS_RISK_OFF     — multi-feature crisis signal; block all new buys immediately
  HIGH_VOL_DOWNTREND  — elevated fear + price weakness; cut size sharply
  DEFENSIVE_DOWNTREND — steady bearish drift; modest size reduction
  BULL_TREND          — confirmed uptrend; normal sizing
  NEUTRAL_CHOP        — directionless; conservative sizing, prefer mean-reversion
  UNKNOWN             — insufficient data; treat as STRESS for safety

Hysteresis:
  STRESS_RISK_OFF is entered immediately (no confirmation delay).
  All other regime transitions require 2 consecutive bars before confirming,
  preventing oscillation on borderline days.

State persistence:
  logs/regime_state.json  — survives restarts; required for hysteresis to work.

Live data cache:
  logs/spy_vix_cache.pkl  — refreshed once per calendar day; avoids repeated
  yfinance round-trips during multi-run days.
"""

import json
import logging
import os
import pickle
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum

import pandas as pd
import yfinance as yf

import config as cfg

logger = logging.getLogger(__name__)

_CACHE_PATH = os.path.join(cfg.LOG_DIR, "spy_vix_cache.pkl")
_STATE_PATH = os.path.join(cfg.LOG_DIR, "regime_state.json")

_MIN_BARS_FULL = 200
_MIN_BARS_PARTIAL = 22
_MIN_BARS_MINIMAL = 6


class MarketRegime(StrEnum):
    STRESS_RISK_OFF = "STRESS_RISK_OFF"
    HIGH_VOL_DOWNTREND = "HIGH_VOL_DOWNTREND"
    DEFENSIVE_DOWNTREND = "DEFENSIVE_DOWNTREND"
    BULL_TREND = "BULL_TREND"
    NEUTRAL_CHOP = "NEUTRAL_CHOP"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class RegimeThresholds:
    # STRESS_RISK_OFF: multi-feature convergence required; single-day drop alone is not enough
    spy_bear_1d: float = -1.8
    spy_5d_stress: float = -5.0
    spy_drawdown_stress: float = -8.0
    vix_stress_absolute: float = 30.0
    vix_spike_ratio: float = 1.4
    vix_5d_surge: float = 30.0
    # HIGH_VOL_DOWNTREND
    vix_high_vol: float = 25.0
    spy_5d_high_vol: float = -3.0
    # DEFENSIVE_DOWNTREND
    spy_5d_defensive: float = -1.5
    # BULL_TREND
    spy_5d_bull: float = 2.0
    spy_1d_bull: float = 0.0
    require_above_ma200: bool = True


@dataclass(frozen=True)
class RegimeFeatures:
    spy_ret_1d: float
    spy_ret_5d: float
    spy_ret_20d: float
    spy_above_ma200: bool | None
    spy_drawdown_pct: float
    vix: float | None
    vix_ma20: float | None
    vix_vs_ma: float | None
    vix_5d_change: float | None
    data_quality: str  # "full" | "partial" | "minimal" | "insufficient"


@dataclass(frozen=True)
class PreviousRegimeState:
    """Persisted after each classification to enable 2-bar hysteresis."""

    regime: MarketRegime
    pending_candidate: MarketRegime | None = None
    pending_count: int = 0


@dataclass(frozen=True)
class MarketRegimeSnapshot:
    """Output of a single regime classification including features and hysteresis state."""

    regime: MarketRegime
    reasons: tuple[str, ...]
    features: RegimeFeatures
    pending_candidate: MarketRegime | None = None
    pending_count: int = 0

    def to_previous(self) -> PreviousRegimeState:
        return PreviousRegimeState(
            regime=self.regime,
            pending_candidate=self.pending_candidate,
            pending_count=self.pending_count,
        )

    def to_dict(self) -> dict:
        """Backward-compatible dict for all existing callers.

        Preserves is_bearish, spy_change_pct, spy_5d_pct, regime (old keys)
        and adds vix, data_quality, reasons (new keys).
        """
        return {
            "is_bearish": self.regime in (MarketRegime.STRESS_RISK_OFF, MarketRegime.UNKNOWN),
            "spy_change_pct": round(self.features.spy_ret_1d, 2),
            "spy_5d_pct": round(self.features.spy_ret_5d, 2),
            "regime": self.regime.value,
            "vix": self.features.vix,
            "data_quality": self.features.data_quality,
            "reasons": list(self.reasons),
        }


def compute_regime_features(
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame | None,
    as_of: str | pd.Timestamp | None = None,
) -> RegimeFeatures:
    """Compute all regime features from SPY and VIX DataFrames up to as_of.

    spy_df must have a 'Close' column indexed by Timestamp.
    vix_df (optional) must have a 'Close' column indexed by Timestamp.
    """
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        spy_df = spy_df[spy_df.index <= cutoff]
        if vix_df is not None:
            vix_df = vix_df[vix_df.index <= cutoff]

    spy_close = spy_df["Close"].dropna()
    n = len(spy_close)

    _insufficient = RegimeFeatures(
        spy_ret_1d=0.0,
        spy_ret_5d=0.0,
        spy_ret_20d=0.0,
        spy_above_ma200=None,
        spy_drawdown_pct=0.0,
        vix=None,
        vix_ma20=None,
        vix_vs_ma=None,
        vix_5d_change=None,
        data_quality="insufficient",
    )
    if n < _MIN_BARS_MINIMAL:
        return _insufficient

    spy_ret_1d = float((spy_close.iloc[-1] / spy_close.iloc[-2] - 1) * 100)
    spy_ret_5d = float((spy_close.iloc[-1] / spy_close.iloc[-6] - 1) * 100) if n >= 6 else 0.0
    spy_ret_20d = float((spy_close.iloc[-1] / spy_close.iloc[-21] - 1) * 100) if n >= 21 else 0.0

    spy_above_ma200: bool | None = None
    if n >= _MIN_BARS_FULL:
        ma200 = float(spy_close.rolling(200).mean().iloc[-1])
        spy_above_ma200 = bool(spy_close.iloc[-1] > ma200)

    window = min(252, n)
    high_52w = float(spy_close.iloc[-window:].max())
    spy_drawdown_pct = float((spy_close.iloc[-1] / high_52w - 1) * 100) if high_52w > 0 else 0.0

    if n >= _MIN_BARS_FULL:
        data_quality = "full"
    elif n >= _MIN_BARS_PARTIAL:
        data_quality = "partial"
    else:
        data_quality = "minimal"

    vix: float | None = None
    vix_ma20: float | None = None
    vix_vs_ma: float | None = None
    vix_5d_change: float | None = None

    if vix_df is not None:
        vix_close = vix_df["Close"].dropna()
        if not vix_close.empty:
            vix = float(vix_close.iloc[-1])
            if len(vix_close) >= 20:
                _vma = float(vix_close.rolling(20).mean().iloc[-1])
                vix_ma20 = _vma
                if _vma > 0:  # pragma: no branch — VIX MA is always positive
                    vix_vs_ma = vix / _vma
            if len(vix_close) >= 6:
                vix_5d_change = float((vix_close.iloc[-1] / vix_close.iloc[-6] - 1) * 100)

    return RegimeFeatures(
        spy_ret_1d=spy_ret_1d,
        spy_ret_5d=spy_ret_5d,
        spy_ret_20d=spy_ret_20d,
        spy_above_ma200=spy_above_ma200,
        spy_drawdown_pct=spy_drawdown_pct,
        vix=vix,
        vix_ma20=vix_ma20,
        vix_vs_ma=vix_vs_ma,
        vix_5d_change=vix_5d_change,
        data_quality=data_quality,
    )


def resolve_regime(
    features: RegimeFeatures,
    thresholds: RegimeThresholds | None = None,
) -> tuple[MarketRegime, list[str]]:
    """Priority-ordered regime resolver.  Returns (regime, reasons).

    STRESS_RISK_OFF requires multiple features converging — a single bad day
    is insufficient.  This prevents false alarms on isolated sell-offs.
    """
    if thresholds is None:
        thresholds = RegimeThresholds()

    t = thresholds
    f = features
    reasons: list[str] = []

    if f.data_quality == "insufficient":
        return MarketRegime.UNKNOWN, ["insufficient SPY data (< 6 bars)"]

    # ── STRESS_RISK_OFF: requires convergence of ≥2 risk dimensions ──────────
    # Trigger A: single-day shock + sustained 5-day weakness
    trigger_a = f.spy_ret_1d <= t.spy_bear_1d and f.spy_ret_5d <= t.spy_5d_stress
    # Trigger B: single-day shock + deep drawdown + elevated VIX
    trigger_b = (
        f.spy_ret_1d <= t.spy_bear_1d
        and f.spy_drawdown_pct <= t.spy_drawdown_stress
        and (f.vix or 0.0) >= t.vix_high_vol
    )
    # Trigger C: single-day shock + absolute VIX fear level
    trigger_c = f.spy_ret_1d <= t.spy_bear_1d and (f.vix or 0.0) >= t.vix_stress_absolute
    # Trigger D: single-day shock + VIX surged hard this week
    trigger_d = (
        f.spy_ret_1d <= t.spy_bear_1d
        and f.vix_5d_change is not None
        and f.vix_5d_change >= t.vix_5d_surge
    )
    # Trigger E: no single-day shock required — sustained 5d weakness + extreme absolute VIX
    trigger_e = f.spy_ret_5d <= t.spy_5d_stress and (f.vix or 0.0) >= t.vix_stress_absolute
    # Trigger F: sustained 5d weakness + VIX spiked far above its moving average
    trigger_f = (
        f.spy_ret_5d <= t.spy_5d_stress
        and f.vix_vs_ma is not None
        and f.vix_vs_ma >= t.vix_spike_ratio
    )

    if trigger_a:
        reasons.append(f"STRESS-A: SPY {f.spy_ret_1d:+.1f}%/1d + {f.spy_ret_5d:+.1f}%/5d")
    if trigger_b:
        reasons.append(
            f"STRESS-B: SPY {f.spy_ret_1d:+.1f}%/1d + drawdown {f.spy_drawdown_pct:.1f}% + VIX {f.vix:.0f}"
        )
    if trigger_c:
        reasons.append(
            f"STRESS-C: SPY {f.spy_ret_1d:+.1f}%/1d + VIX {f.vix:.0f} ≥ {t.vix_stress_absolute:.0f}"
        )
    if trigger_d:
        reasons.append(
            f"STRESS-D: SPY {f.spy_ret_1d:+.1f}%/1d + VIX +{f.vix_5d_change:.0f}%/5d spike"
        )
    if trigger_e:
        reasons.append(
            f"STRESS-E: SPY {f.spy_ret_5d:+.1f}%/5d sustained + VIX {f.vix:.0f} ≥ {t.vix_stress_absolute:.0f}"
        )
    if trigger_f:
        reasons.append(
            f"STRESS-F: SPY {f.spy_ret_5d:+.1f}%/5d sustained + VIX/MA {f.vix_vs_ma:.2f} ≥ {t.vix_spike_ratio}"
        )

    if any([trigger_a, trigger_b, trigger_c, trigger_d, trigger_e, trigger_f]):
        return MarketRegime.STRESS_RISK_OFF, reasons

    # ── HIGH_VOL_DOWNTREND ───────────────────────────────────────────────────
    if f.vix is not None and f.vix >= t.vix_high_vol and f.spy_ret_5d <= t.spy_5d_high_vol:
        reasons.append(f"VIX {f.vix:.1f} ≥ {t.vix_high_vol} + SPY {f.spy_ret_5d:+.1f}%/5d")
        return MarketRegime.HIGH_VOL_DOWNTREND, reasons

    # ── DEFENSIVE_DOWNTREND ──────────────────────────────────────────────────
    if f.spy_ret_5d <= t.spy_5d_defensive:
        reasons.append(f"SPY {f.spy_ret_5d:+.1f}%/5d — steady weakness")
        return MarketRegime.DEFENSIVE_DOWNTREND, reasons

    # ── BULL_TREND ───────────────────────────────────────────────────────────
    bull_price_ok = f.spy_ret_5d >= t.spy_5d_bull and f.spy_ret_1d >= t.spy_1d_bull
    ma200_ok = (not t.require_above_ma200) or (f.spy_above_ma200 is True)
    if bull_price_ok and ma200_ok:
        ma200_str = " + above MA200" if t.require_above_ma200 else ""
        reasons.append(f"SPY {f.spy_ret_5d:+.1f}%/5d + {f.spy_ret_1d:+.1f}%/1d{ma200_str}")
        return MarketRegime.BULL_TREND, reasons

    # ── NEUTRAL_CHOP ─────────────────────────────────────────────────────────
    reasons.append(f"SPY {f.spy_ret_1d:+.1f}%/1d  {f.spy_ret_5d:+.1f}%/5d — no directional signal")
    return MarketRegime.NEUTRAL_CHOP, reasons


def apply_regime_hysteresis(
    candidate: MarketRegime,
    previous: PreviousRegimeState | None,
    reasons: list[str],
    features: RegimeFeatures,
) -> MarketRegimeSnapshot:
    """Apply 2-bar confirmation hysteresis to prevent daily flip-flopping.

    STRESS_RISK_OFF and UNKNOWN confirm immediately (safety-first).
    All other regime transitions require 2 consecutive bars of the candidate
    before taking effect.
    """
    # STRESS / UNKNOWN / first-run: confirm immediately
    if previous is None or candidate in (MarketRegime.STRESS_RISK_OFF, MarketRegime.UNKNOWN):
        return MarketRegimeSnapshot(
            regime=candidate,
            reasons=tuple(reasons),
            features=features,
            pending_candidate=None,
            pending_count=0,
        )

    current = previous.regime

    if candidate == current:
        return MarketRegimeSnapshot(
            regime=current,
            reasons=tuple(reasons),
            features=features,
            pending_candidate=None,
            pending_count=0,
        )

    # Count consecutive bars for this candidate
    if previous.pending_candidate == candidate:
        new_count = previous.pending_count + 1
    else:
        new_count = 1

    if new_count >= 2:
        return MarketRegimeSnapshot(
            regime=candidate,
            reasons=tuple(reasons),
            features=features,
            pending_candidate=None,
            pending_count=0,
        )

    # First bar of the candidate — hold current, record pending
    hold_reasons = (
        f"[hysteresis: holding {current.value}; {candidate.value} needs 1 more bar]",
        *reasons,
    )
    return MarketRegimeSnapshot(
        regime=current,
        reasons=hold_reasons,
        features=features,
        pending_candidate=candidate,
        pending_count=new_count,
    )


def get_market_regime(
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame | None,
    as_of: str | pd.Timestamp | None = None,
    previous: PreviousRegimeState | None = None,
    thresholds: RegimeThresholds | None = None,
) -> MarketRegimeSnapshot:
    """Full regime classification: features → resolution → hysteresis."""
    features = compute_regime_features(spy_df, vix_df, as_of)
    candidate, reasons = resolve_regime(features, thresholds)
    return apply_regime_hysteresis(candidate, previous, reasons, features)


def compute_regime_series(
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame | None,
    trading_dates: list[str],
    thresholds: RegimeThresholds | None = None,
) -> dict[str, str]:
    """Classify each date in trading_dates using rolling history.

    Returns {date_str: regime_name_str} for use by the backtest engine.
    Hysteresis is applied sequentially across sorted dates.
    """
    previous: PreviousRegimeState | None = None
    result: dict[str, str] = {}
    for date_str in sorted(trading_dates):
        snapshot = get_market_regime(
            spy_df, vix_df, as_of=date_str, previous=previous, thresholds=thresholds
        )
        result[date_str] = snapshot.regime.value
        previous = snapshot.to_previous()
    return result


# ── State persistence ────────────────────────────────────────────────────────


def load_regime_state() -> PreviousRegimeState | None:
    """Load persisted regime state from disk.  Returns None on any failure."""
    try:
        with open(_STATE_PATH) as f:
            data = json.load(f)
        regime = MarketRegime(data["regime"])
        pending_raw = data.get("pending_candidate")
        pending = MarketRegime(pending_raw) if pending_raw else None
        return PreviousRegimeState(
            regime=regime,
            pending_candidate=pending,
            pending_count=int(data.get("pending_count", 0)),
        )
    except FileNotFoundError:
        return None
    except (KeyError, ValueError) as e:
        logger.warning(f"Regime state file corrupt, ignoring: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to load regime state: {e}")
        return None


def save_regime_state(snapshot: MarketRegimeSnapshot) -> None:
    """Persist regime state for hysteresis continuity across restarts."""
    try:
        os.makedirs(cfg.LOG_DIR, exist_ok=True)
        data = {
            "regime": snapshot.regime.value,
            "pending_candidate": (
                snapshot.pending_candidate.value if snapshot.pending_candidate else None
            ),
            "pending_count": snapshot.pending_count,
            "saved_at": datetime.utcnow().isoformat(),
        }
        with open(_STATE_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save regime state: {e}")


# ── Live data caching ────────────────────────────────────────────────────────


def _load_cache() -> tuple[pd.DataFrame | None, pd.DataFrame | None, date | None]:
    """Return (spy_df, vix_df, cache_date) or (None, None, None) on miss."""
    try:
        with open(_CACHE_PATH, "rb") as f:
            payload = pickle.load(f)
        return payload["spy"], payload["vix"], payload["date"]
    except Exception:
        return None, None, None


def _save_cache(spy_df: pd.DataFrame, vix_df: pd.DataFrame | None) -> None:
    try:
        os.makedirs(cfg.LOG_DIR, exist_ok=True)
        with open(_CACHE_PATH, "wb") as f:
            pickle.dump({"spy": spy_df, "vix": vix_df, "date": date.today()}, f)
    except Exception as e:
        logger.warning(f"Failed to save SPY/VIX cache: {e}")


def fetch_spy_vix_history(
    lookback_days: int = 504,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Fetch SPY and VIX history with a daily pickle cache.

    Returns (spy_df, vix_df) where each has a tz-naive DatetimeIndex and a
    'Close' column.  vix_df is None if the VIX fetch fails.

    lookback_days=504 ≈ 2 calendar years, giving ~500 trading days — enough
    for MA200 plus a runway to avoid cold-start gaps.
    """
    spy_cached, vix_cached, cache_date = _load_cache()
    if cache_date == date.today() and spy_cached is not None:
        logger.debug("SPY/VIX cache hit — skipping yfinance download")
        return spy_cached, vix_cached

    start = (date.today() - timedelta(days=lookback_days)).isoformat()

    spy_df: pd.DataFrame
    try:
        raw = yf.download("SPY", start=start, auto_adjust=False, progress=False)
        if raw.empty:
            raise ValueError("yfinance returned empty SPY DataFrame")
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        spy_df = pd.DataFrame({"Close": close.astype(float)}).dropna()
        spy_df.index = pd.DatetimeIndex(spy_df.index).tz_localize(None)
    except Exception as e:
        logger.error(f"SPY history fetch failed: {e}")
        if spy_cached is not None:
            logger.warning("Using stale SPY cache after fetch failure")
            return spy_cached, vix_cached
        return pd.DataFrame(columns=["Close"]), None

    vix_df: pd.DataFrame | None = None
    try:
        raw_vix = yf.download("^VIX", start=start, auto_adjust=False, progress=False)
        if not raw_vix.empty:
            vix_close = raw_vix["Close"]
            if isinstance(vix_close, pd.DataFrame):
                vix_close = vix_close.iloc[:, 0]
            vix_df = pd.DataFrame({"Close": vix_close.astype(float)}).dropna()
            vix_df.index = pd.DatetimeIndex(vix_df.index).tz_localize(None)
    except Exception as e:
        logger.warning(f"VIX history fetch failed: {e}")

    _save_cache(spy_df, vix_df)
    return spy_df, vix_df
