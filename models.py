from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class BrokerStateUnavailable(Exception):
    """Raised when a broker query fails and the result cannot be trusted.

    Any code path that catches this in the buy loop must block the buy — treating
    broker uncertainty as permissive is the most dangerous failure mode.
    """


class OrderLedgerUnavailable(Exception):
    """Raised when the order-intent ledger cannot be queried.

    Any code path that catches this in the buy loop must block the buy — a ledger
    failure means the restart-safe duplicate-order guard is inoperative.
    """


# ── Execution result types ────────────────────────────────────────────────────


class OrderStatus(StrEnum):
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    TIMEOUT = "TIMEOUT"
    REJECTED = "REJECTED"
    STOP_FAILED = "STOP_FAILED"
    UNPROTECTED = "UNPROTECTED"


@dataclass
class OrderResult:
    status: OrderStatus
    symbol: str
    filled_qty: float = 0.0
    filled_avg_price: float = 0.0
    broker_order_id: str | None = None
    rejection_reason: str | None = None
    stop_order_id: str | None = None

    @property
    def is_success(self) -> bool:
        return self.status == OrderStatus.FILLED


VALID_BUY_SIGNALS: frozenset[str] = frozenset(
    {
        "mean_reversion",
        "momentum",
        "trend_continuation",
        "macd_crossover",
        "rsi_oversold",
        "news_catalyst",
        "unknown",
        "bb_squeeze",
        "breakout_52w",
        "rs_leader",
        "inside_day_breakout",
        "trend_pullback",
        "vwap_reclaim",
        "orb_breakout",
        "intraday_momentum",
        "gap_and_go",
        "vix_fear_reversion",
    }
)


class BuyCandidate(BaseModel):
    symbol: str
    confidence: int = Field(ge=1, le=10)
    reasoning: str = Field(min_length=20, max_length=2000)
    key_signal: str | None = None
    do_nothing_case: str = Field(min_length=10, max_length=500)
    invalidation_trigger: str = Field(min_length=10, max_length=500)

    model_config = {"extra": "ignore"}

    @field_validator("key_signal")
    @classmethod
    def validate_signal(cls, v: str | None) -> str | None:
        if v is not None and v not in VALID_BUY_SIGNALS:
            raise ValueError(f"Unknown signal '{v}'")
        return v


class PositionDecision(BaseModel):
    symbol: str
    action: Literal["HOLD", "SELL"]
    reasoning: str = ""

    model_config = {"extra": "ignore"}


class DecisionSet(BaseModel):
    market_summary: str = Field(min_length=10, max_length=300)
    position_decisions: list[PositionDecision]
    buy_candidates: list[BuyCandidate] = Field(default_factory=list)

    model_config = {"extra": "ignore"}

    @model_validator(mode="after")
    def no_duplicate_buy_symbols(self) -> DecisionSet:
        symbols = [c.symbol for c in self.buy_candidates]
        seen: set[str] = set()
        duplicates: set[str] = set()
        for s in symbols:
            if s in seen:
                duplicates.add(s)
            seen.add(s)
        if duplicates:
            raise ValueError(f"Duplicate buy candidates: {sorted(duplicates)}")
        return self

    @model_validator(mode="after")
    def no_buy_sell_conflict(self) -> DecisionSet:
        buy_symbols = {c.symbol for c in self.buy_candidates}
        sell_symbols = {d.symbol for d in self.position_decisions if d.action == "SELL"}
        conflicts = buy_symbols & sell_symbols
        if conflicts:
            raise ValueError(
                f"Symbol(s) in both BUY candidates and SELL decisions: {sorted(conflicts)}"
            )
        return self
