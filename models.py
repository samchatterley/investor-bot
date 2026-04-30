from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

VALID_BUY_SIGNALS: frozenset[str] = frozenset({
    "mean_reversion",
    "momentum",
    "trend_continuation",
    "macd_crossover",
    "rsi_oversold",
    "news_catalyst",
    "unknown",
})


class BuyCandidate(BaseModel):
    symbol: str
    confidence: int = Field(ge=1, le=10)
    reasoning: str = Field(min_length=20, max_length=500)
    key_signal: str | None = None

    model_config = {"extra": "allow"}

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

    model_config = {"extra": "allow"}


class DecisionSet(BaseModel):
    market_summary: str = Field(min_length=10, max_length=300)
    position_decisions: list[PositionDecision]
    buy_candidates: list[BuyCandidate]

    model_config = {"extra": "allow"}

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
        sell_symbols = {
            d.symbol for d in self.position_decisions if d.action == "SELL"
        }
        conflicts = buy_symbols & sell_symbols
        if conflicts:
            raise ValueError(
                f"Symbol(s) in both BUY candidates and SELL decisions: {sorted(conflicts)}"
            )
        return self
