from config import STOP_LOSS_PCT, TAKE_PROFIT_PCT, MAX_POSITION_PCT, KELLY_MULTIPLIER


def kelly_fraction(confidence: int) -> float:
    """
    Half-Kelly position sizing based on AI confidence score (1-10).

    Kelly formula: f = (p*b - q) / b
      p = probability of winning (confidence / 10)
      q = 1 - p
      b = reward/risk ratio (take_profit / stop_loss)

    We use half-Kelly (multiplied by KELLY_MULTIPLIER) to reduce variance
    and cap at MAX_POSITION_PCT to prevent over-concentration.

    Example outputs (take_profit=12%, stop_loss=4%):
      confidence 7  → ~28% of available cash
      confidence 8  → ~35%
      confidence 9  → ~43%
      confidence 10 → capped at MAX_POSITION_PCT
    """
    p = confidence / 10.0
    q = 1.0 - p
    b = TAKE_PROFIT_PCT / max(STOP_LOSS_PCT, 1e-6)  # reward-to-risk ratio
    raw_kelly = (p * b - q) / b
    fraction = max(0.0, raw_kelly * KELLY_MULTIPLIER)
    return min(fraction, MAX_POSITION_PCT)


def get_max_positions(portfolio_value: float) -> int:
    """
    Gradually unlock more position slots as the account grows.
    Keeps individual positions large enough to matter.
    """
    if portfolio_value >= 50000:
        return 5
    elif portfolio_value >= 20000:
        return 4
    else:
        return 3
