from datetime import date

from config import today_et

# High-risk scheduled macro events for 2026.
# On these days, volatility is structurally elevated and unpredictable —
# we skip new buys but still manage existing positions.
#
# Sources: Federal Reserve calendar, BLS CPI schedule, BLS NFP schedule.

FOMC_ANNOUNCEMENT_DATES = {
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 5, 6),
    date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
    date(2026, 10, 28), date(2026, 12, 9),
}

CPI_RELEASE_DATES = {
    date(2026, 1, 15), date(2026, 2, 12), date(2026, 3, 12),
    date(2026, 4, 10), date(2026, 5, 14), date(2026, 6, 11),
    date(2026, 7, 15), date(2026, 8, 13), date(2026, 9, 10),
    date(2026, 10, 9), date(2026, 11, 12), date(2026, 12, 10),
}

NFP_RELEASE_DATES = {
    date(2026, 1, 9), date(2026, 2, 6), date(2026, 3, 6),
    date(2026, 4, 3), date(2026, 5, 1), date(2026, 6, 5),
    date(2026, 7, 10), date(2026, 8, 7), date(2026, 9, 4),
    date(2026, 10, 2), date(2026, 11, 6), date(2026, 12, 4),
}

EVENT_LABELS = {
    **dict.fromkeys(FOMC_ANNOUNCEMENT_DATES, "FOMC rate decision"),
    **dict.fromkeys(CPI_RELEASE_DATES, "CPI inflation release"),
    **dict.fromkeys(NFP_RELEASE_DATES, "Non-Farm Payrolls release"),
}


def get_macro_risk(check_date: date | None = None) -> dict:
    """
    Returns {'is_high_risk': bool, 'event': str | None}.
    High-risk days: FOMC announcements, CPI releases, NFP releases.
    """
    if check_date is None:
        check_date = today_et()

    all_risk_dates = FOMC_ANNOUNCEMENT_DATES | CPI_RELEASE_DATES | NFP_RELEASE_DATES

    if check_date in all_risk_dates:
        return {"is_high_risk": True, "event": EVENT_LABELS.get(check_date, "Macro event")}

    return {"is_high_risk": False, "event": None}
