"""Point-in-time feature/outcome dataset: the rigour-critical core (EXPERIMENT.md sections 4, 15.7).

This module holds the *methodology* that makes the dataset faithful, all pure and lookahead-clean:

  - AsOfExpectancy: signal / signal-by-regime edge computed strictly from outcomes that had already
    *closed* before the decision (never merely entered, never full-sample). It exposes BOTH an
    expanding-window and a rolling-window estimate plus a decay term, so the fitted v2 baseline learns
    the window weighting from data rather than the researcher hand-picking it (the faithful choice).
  - forward_r: path-independent H-day forward return, ATR-normalised, net of costs (in R units).
  - blind_features: the de-identified view fed to the LLM Arm 2 historically (no ticker / date /
    absolute price), per the v1.3 knowledge-cutoff blinding rule.
  - split_tag: train / validation / holdout by date (holdout = 2024-01-01+).

The engine-wired orchestration (build_dataset, which reuses backtest.engine's _compute_indicators,
_row_to_snapshot, _entry_signal, RS ranks, regime map, and cost model over historical prices) is the
follow-up; it depends only on these primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_TRAIN_END = "2022-12-31"
_VALIDATION_END = "2023-12-31"  # holdout = 2024-01-01 onward (EXPERIMENT.md v1.2/v1.3)
_DEFAULT_DECAY_WINDOW = 250  # rolling window (recent closed outcomes) for the decay term
# Fields stripped from the LLM's blinded view so it cannot identify the security or period.
_BLIND_DROP = frozenset({"symbol", "date", "current_price", "Close"})


def forward_r(
    closes: list[float],
    entry_index: int,
    horizon: int,
    atr: float,
    cost_r: float = 0.0,
    direction: str = "long",
) -> float | None:
    """Path-independent H-day forward return in ATR units, net of cost (already in R units).

    Entry at the decision bar's close (closes[entry_index]); exit at the close `horizon` bars later.
    Returns None when ATR is non-positive or the exit bar is beyond the series.
    """
    if atr <= 0:
        return None
    exit_index = entry_index + horizon
    if entry_index < 0 or exit_index >= len(closes):
        return None
    entry = closes[entry_index]
    exit_price = closes[exit_index]
    raw = (exit_price - entry) if direction == "long" else (entry - exit_price)
    return raw / atr - cost_r


@dataclass
class _Outcome:
    known_at: (
        str  # ISO date the outcome became known (entry date + horizon); used for point-in-time
    )
    signal: str
    regime: str
    r: float


class AsOfExpectancy:
    """Point-in-time signal expectancy from outcomes that closed strictly before the decision date.

    Faithful design: exposes expanding and rolling edges plus a decay term as separate features, so the
    fitted baseline (v2) learns how to weight long-run vs recent edge instead of the researcher fixing
    a window. All queries use only outcomes with ``known_at < decision_date``.
    """

    def __init__(self, decay_window: int = _DEFAULT_DECAY_WINDOW) -> None:
        self._outcomes: list[_Outcome] = []
        self._decay_window = decay_window

    def record(self, known_at: str, signal: str, regime: str, r: float) -> None:
        self._outcomes.append(_Outcome(known_at=known_at, signal=signal, regime=regime, r=r))

    def _prior(self, signal: str, decision_date: str, regime: str | None = None) -> list[float]:
        return [
            o.r
            for o in self._outcomes
            if o.known_at < decision_date
            and o.signal == signal
            and (regime is None or o.regime == regime)
        ]

    def expanding_edge(self, signal: str, decision_date: str) -> float | None:
        rs = self._prior(signal, decision_date)
        return sum(rs) / len(rs) if rs else None

    def regime_edge(self, signal: str, regime: str, decision_date: str) -> float | None:
        rs = self._prior(signal, decision_date, regime=regime)
        return sum(rs) / len(rs) if rs else None

    def rolling_edge(
        self, signal: str, decision_date: str, window: int | None = None
    ) -> float | None:
        window = window or self._decay_window
        rs = self._prior(signal, decision_date)
        if not rs:
            return None
        recent = rs[-window:]
        return sum(recent) / len(recent)

    def decay(self, signal: str, decision_date: str, window: int | None = None) -> float | None:
        """Recent edge minus long-run edge (negative => the signal is decaying)."""
        expanding = self.expanding_edge(signal, decision_date)
        rolling = self.rolling_edge(signal, decision_date, window)
        if expanding is None or rolling is None:
            return None
        return rolling - expanding

    def features(self, signal: str, regime: str, decision_date: str) -> dict[str, float | None]:
        return {
            "signal_edge_expanding": self.expanding_edge(signal, decision_date),
            "signal_edge_rolling": self.rolling_edge(signal, decision_date),
            "signal_regime_edge": self.regime_edge(signal, regime, decision_date),
            "signal_decay": self.decay(signal, decision_date),
        }


def blind_features(features: dict, drop: frozenset[str] = _BLIND_DROP) -> dict:
    """De-identified feature view for the LLM Arm 2 (drops ticker/date/absolute-price fields)."""
    return {k: v for k, v in features.items() if k not in drop and not k.endswith("_price")}


def split_tag(
    decision_date: str,
    train_end: str = _TRAIN_END,
    validation_end: str = _VALIDATION_END,
) -> str:
    """Map an ISO decision date to its experiment split (holdout = 2024-01-01 onward)."""
    if decision_date <= train_end:
        return "train"
    if decision_date <= validation_end:
        return "validation"
    return "holdout"


@dataclass
class DatasetRow:
    symbol: str
    date: str
    split: str
    features: dict
    blinded_features: dict
    fired_signals: list[str]
    material_context: list[str]
    expectancy: dict
    forward_r: float | None = None
    extra: dict = field(default_factory=dict)


def assemble_row(
    symbol: str,
    date: str,
    features: dict,
    fired_signals: list[str],
    material_context: list[str],
    expectancy: dict,
    forward_r_value: float | None,
    train_end: str = _TRAIN_END,
    validation_end: str = _VALIDATION_END,
) -> DatasetRow:
    """Package one point-in-time observation, computing its blinded view and split tag."""
    full = {**features, "symbol": symbol, "date": date}
    return DatasetRow(
        symbol=symbol,
        date=date,
        split=split_tag(date, train_end, validation_end),
        features=features,
        blinded_features=blind_features(full),
        fired_signals=fired_signals,
        material_context=material_context,
        expectancy=expectancy,
        forward_r=forward_r_value,
    )
