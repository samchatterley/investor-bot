import smtplib
import logging
from datetime import date
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config import EMAIL_FROM, EMAIL_TO, EMAIL_CC, EMAIL_APP_PASSWORD, EMAIL_RECIPIENTS

logger = logging.getLogger(__name__)


def _all_recipients() -> list[str]:
    """Return flat recipient list (legacy fallback — no names)."""
    recipients = [EMAIL_TO] if EMAIL_TO else []
    if EMAIL_CC:
        recipients += [a.strip() for a in EMAIL_CC.split(",") if a.strip()]
    return recipients


def _named_recipients() -> list[tuple[str, str]]:
    """Return [(first_name, email), ...].
    Reads EMAIL_RECIPIENTS ("Name:email,...") if set, otherwise falls back
    to EMAIL_TO + EMAIL_CC with a generic greeting.
    """
    if EMAIL_RECIPIENTS:
        result = []
        for part in EMAIL_RECIPIENTS.split(","):
            part = part.strip()
            if ":" in part:
                name, email = part.split(":", 1)
                result.append((name.strip(), email.strip()))
            elif part:
                result.append(("there", part))
        return result
    return [("there", e) for e in _all_recipients()]


def _send_html(subject: str, html_fn):
    """Send a personalised HTML email to each recipient individually.
    html_fn: callable(first_name: str) -> str
    """
    if not EMAIL_FROM or not EMAIL_APP_PASSWORD:
        logger.warning("Email not configured — skipping. Add EMAIL_FROM and EMAIL_APP_PASSWORD to .env")
        return
    recipients = _named_recipients()
    if not recipients:
        logger.warning("No recipients configured — skipping. Add EMAIL_RECIPIENTS to .env")
        return
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
            for name, email in recipients:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = subject
                msg["From"] = EMAIL_FROM
                msg["To"] = email
                msg.attach(MIMEText(html_fn(name), "html"))
                server.sendmail(EMAIL_FROM, [email], msg.as_string())
                logger.info(f"Email sent to {email} ({name})")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")

_SIGNAL_LABELS = {
    "mean_reversion":      "Oversold bounce",
    "momentum":            "Upward momentum",
    "trend_continuation":  "Continuing uptrend",
    "macd_crossover":      "Momentum shift",
    "rsi_oversold":        "Oversold reversal",
    "news_catalyst":       "News-driven move",
    "unknown":             "Mixed signals",
}

_GLOSSARY = [
    ("Kelly sizing",          "A maths-based formula that decides how much to invest in each trade — the more confident the signal, the more capital is deployed."),
    ("Confidence score",      "The bot's self-assessed certainty that a trade will work, scored 1–10. Only scores of 7 or above trigger a trade."),
    ("Oversold bounce",       "The stock has fallen further than expected and looks likely to recover — the bot buys in anticipation of that recovery."),
    ("Upward momentum",       "The stock is trending strongly upward with above-average trading volume, suggesting the move has more to run."),
    ("Continuing uptrend",    "The stock is already in an uptrend and the signals suggest it will keep climbing in the short term."),
    ("Momentum shift",        "A technical indicator just flipped from negative to positive, suggesting the stock is turning a corner."),
    ("Trailing stop",         "A stop-loss that automatically moves up as the stock rises — it locks in gains while still giving the stock room to grow."),
    ("Partial exit",          "The bot sold half the position after reaching the profit target, locking in gains while keeping skin in the game."),
    ("Circuit breaker",       "A safety rule: if the overall portfolio falls too much in a short period, the bot stops making new trades until things stabilise."),
    ("Bear market filter",    "If the broader market drops sharply in a day, the bot skips new buys — it's better to sit out than buy into a falling market."),
    ("Earnings exit",         "Companies report earnings quarterly, which can cause big unexpected price swings. The bot exits before earnings to avoid that risk."),
]


def _humanise_detail(detail: str) -> str:
    """Convert the raw detail string into plain English for the email."""
    if not detail or detail == "dry run":
        return "Simulated — no real order placed"

    parts = [p.strip() for p in detail.split("|")]
    result = []
    for part in parts:
        if part.startswith("$") and not part.startswith("$0"):
            result.append(f"<b>Invested {part}</b>")
        elif "Kelly" in part:
            pct = part.replace("Kelly", "").strip()
            result.append(f"Sized at {pct} of available cash")
        elif "confidence=" in part:
            val = part.replace("confidence=", "").strip()
            result.append(f"Confidence: {val}/10")
        elif part in _SIGNAL_LABELS:
            result.append(f"Signal: {_SIGNAL_LABELS[part]}")
        elif part.startswith("dry run"):
            result.append("Simulated")
        elif part:
            result.append(part)
    return " &nbsp;·&nbsp; ".join(result)


def _build_trade_cards(record: dict) -> str:
    all_trades = []

    for sl in record.get("stop_losses_triggered", []):
        all_trades.append({
            "symbol": sl["symbol"],
            "action": "STOP LOSS",
            "detail": _humanise_detail(f"Closed at {sl['pl_pct']:.1f}%"),
            "reasoning": "Position hit the trailing stop and was automatically closed to protect capital.",
            "bg": "#fff3e0",
            "badge_bg": "#e65100",
            "reasoning_bg": "#fff8f3",
        })

    for t in record.get("trades_executed", []):
        action = t.get("action", "?")
        reasoning = ""
        for b in record.get("buy_candidates", []):
            if b["symbol"] == t["symbol"]:
                reasoning = b.get("reasoning", "")
                break
        for d in record.get("position_decisions", []):
            if d["symbol"] == t["symbol"] and action == "SELL":
                reasoning = d.get("reasoning", "")
                break

        if action == "BUY":
            bg, badge_bg, reasoning_bg = "#f1f8f1", "#2e7d32", "#f7fbf7"
        else:
            bg, badge_bg, reasoning_bg = "#fff5f5", "#c62828", "#fffafa"

        all_trades.append({
            "symbol": t["symbol"],
            "action": action,
            "detail": _humanise_detail(t.get("detail", "")),
            "reasoning": reasoning,
            "bg": bg,
            "badge_bg": badge_bg,
            "reasoning_bg": reasoning_bg,
        })

    if not all_trades:
        return """<table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:8px">
  <tr><td style="padding:20px;background:#f9f9f9;border-radius:8px;text-align:center;color:#888;font-family:Arial,Helvetica,sans-serif;font-size:14px">
    No trades today &#8212; holding existing positions.
  </td></tr>
</table>"""

    cards = ""
    for t in all_trades:
        detail_row = ""
        if t["detail"]:
            detail_row = f"""
  <tr>
    <td style="padding:6px 16px 14px;background:{t['bg']};font-family:Arial,Helvetica,sans-serif;font-size:12px;color:#777;line-height:1.5">
      {t['detail']}
    </td>
  </tr>"""

        reasoning_row = ""
        if t.get("reasoning"):
            reasoning_row = f"""
  <tr>
    <td style="padding:12px 16px 14px;background:{t['reasoning_bg']};border-top:1px solid #e8e8e8;font-family:Arial,Helvetica,sans-serif">
      <p style="font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:#aaa;margin:0 0 4px 0">Why</p>
      <p style="font-size:13px;color:#666;line-height:1.6;margin:0">{t['reasoning']}</p>
    </td>
  </tr>"""

        cards += f"""<table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:12px;border:1px solid #e0e0e0;border-radius:8px;overflow:hidden">
  <tr>
    <td style="background:{t['bg']};padding:14px 16px 0">
      <span style="font-size:18px;font-weight:bold;color:#111;font-family:Arial,Helvetica,sans-serif">{t['symbol']}</span>
      <span style="margin-left:8px;background:{t['badge_bg']};color:#ffffff;font-size:11px;font-weight:bold;padding:3px 8px;border-radius:4px;text-transform:uppercase;font-family:Arial,Helvetica,sans-serif">{t['action']}</span>
    </td>
  </tr>
  {detail_row}
  {reasoning_row}
</table>
"""
    return cards


def _build_html(record: dict, name: str = "there") -> str:
    pnl = record["daily_pnl"]
    pnl_colour = "#2e7d32" if pnl >= 0 else "#c62828"
    pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
    pnl_bg = "#f1f8f1" if pnl >= 0 else "#fff5f5"

    opening_capital = record["account_before"]["portfolio_value"]
    closing_portfolio = record["account_after"]["portfolio_value"]
    day_return_pct = (pnl / opening_capital * 100) if opening_capital else 0.0
    day_return_str = f"+{day_return_pct:.2f}%" if day_return_pct >= 0 else f"{day_return_pct:.2f}%"

    trade_count = len(record.get("trades_executed", [])) + len(record.get("stop_losses_triggered", []))
    trade_line = f"{trade_count} trade{'s' if trade_count != 1 else ''} today" if trade_count else "No trades today"

    is_paper = "paper" in str(record.get("mode", "paper")).lower()
    mode_label = "Paper trading" if is_paper else "Live trading"
    mode_colour = "#999999" if is_paper else "#e65100"

    trade_cards = _build_trade_cards(record)

    glossary_rows = "".join(
        f'<tr>'
        f'<td style="padding:5px 12px 5px 0;color:#555;font-weight:600;vertical-align:top;white-space:nowrap;font-family:Arial,Helvetica,sans-serif;font-size:12px">{term}</td>'
        f'<td style="padding:5px 0;color:#777;line-height:1.5;font-family:Arial,Helvetica,sans-serif;font-size:12px">{explanation}</td>'
        f'</tr>'
        for term, explanation in _GLOSSARY
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <title>Trading Bot Daily Summary</title>
</head>
<body style="margin:0;padding:0;background:#efefef;-webkit-text-size-adjust:100%;-ms-text-size-adjust:100%">

  <table width="100%" cellpadding="0" cellspacing="0" style="background:#efefef">
    <tr><td align="center" style="padding:24px 12px">

      <!-- Outer card -->
      <table width="100%" cellpadding="0" cellspacing="0" style="max-width:600px;background:#ffffff;border-radius:10px;overflow:hidden">
        <tr><td style="padding:32px 28px 0">

          <p style="font-family:Arial,Helvetica,sans-serif;font-size:16px;color:#555;margin:0 0 4px 0">Hi {name},</p>
          <p style="font-family:Arial,Helvetica,sans-serif;font-size:15px;color:#777;margin:0 0 28px 0">Here&#39;s your trading update for <b>{record['date']}</b>.</p>

          <!-- P&L hero -->
          <table width="100%" cellpadding="0" cellspacing="0" style="background:{pnl_bg};border-radius:10px;margin-bottom:24px">
            <tr><td style="padding:24px;text-align:center">
              <p style="font-family:Arial,Helvetica,sans-serif;font-size:13px;color:#888;margin:0 0 4px 0">Today&#39;s P&amp;L</p>
              <p style="font-family:Arial,Helvetica,sans-serif;font-size:44px;font-weight:bold;color:{pnl_colour};line-height:1;margin:0">{pnl_str}</p>
              <p style="font-family:Arial,Helvetica,sans-serif;font-size:16px;font-weight:600;color:{pnl_colour};margin:6px 0 20px">{day_return_str}</p>

              <!-- Stats 2×2 -->
              <table width="100%" cellpadding="0" cellspacing="0">
                <tr>
                  <td width="50%" style="text-align:center;padding:0 8px 14px 0;border-right:1px solid #dddddd">
                    <p style="font-family:Arial,Helvetica,sans-serif;font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.5px;margin:0 0 4px 0">Opening Capital</p>
                    <p style="font-family:Arial,Helvetica,sans-serif;font-size:17px;font-weight:600;color:#333;margin:0">${opening_capital:,.2f}</p>
                  </td>
                  <td width="50%" style="text-align:center;padding:0 0 14px 8px">
                    <p style="font-family:Arial,Helvetica,sans-serif;font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.5px;margin:0 0 4px 0">Closing Portfolio</p>
                    <p style="font-family:Arial,Helvetica,sans-serif;font-size:17px;font-weight:600;color:#333;margin:0">${closing_portfolio:,.2f}</p>
                  </td>
                </tr>
                <tr>
                  <td width="50%" style="text-align:center;padding:14px 8px 0 0;border-right:1px solid #dddddd;border-top:1px solid #dddddd">
                    <p style="font-family:Arial,Helvetica,sans-serif;font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.5px;margin:0 0 4px 0">Cash</p>
                    <p style="font-family:Arial,Helvetica,sans-serif;font-size:17px;font-weight:600;color:#333;margin:0">${record['account_after']['cash']:,.2f}</p>
                  </td>
                  <td width="50%" style="text-align:center;padding:14px 0 0 8px;border-top:1px solid #dddddd">
                    <p style="font-family:Arial,Helvetica,sans-serif;font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.5px;margin:0 0 4px 0">Trades</p>
                    <p style="font-family:Arial,Helvetica,sans-serif;font-size:17px;font-weight:600;color:#333;margin:0">{trade_count}</p>
                  </td>
                </tr>
              </table>
            </td></tr>
          </table>

          <!-- Market summary -->
          <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:24px;border-radius:0 6px 6px 0;overflow:hidden">
            <tr>
              <td width="4" style="background:#cccccc">&nbsp;</td>
              <td style="background:#f5f5f5;padding:10px 14px;font-family:Arial,Helvetica,sans-serif;font-size:14px;color:#555;line-height:1.5">
                <b>Market:</b> {record.get('market_summary', '')}
              </td>
            </tr>
          </table>

          <!-- Trades section -->
          <p style="font-family:Arial,Helvetica,sans-serif;font-size:15px;font-weight:600;color:#333;margin:0 0 14px 0">{trade_line}</p>

          {trade_cards}

          <!-- Glossary -->
          <table width="100%" cellpadding="0" cellspacing="0" style="margin-top:32px;border-top:1px solid #eeeeee">
            <tr><td style="padding-top:20px">
              <p style="font-family:Arial,Helvetica,sans-serif;font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.5px;margin:0 0 14px 0">Terms explained</p>
              <table width="100%" cellpadding="0" cellspacing="0">
                {glossary_rows}
              </table>
            </td></tr>
          </table>

          <!-- Footer -->
          <p style="font-family:Arial,Helvetica,sans-serif;font-size:11px;color:{mode_colour};margin:24px 0 0;text-align:center;padding-bottom:28px">
            {mode_label} &nbsp;&#183;&nbsp; Your trading bot
          </p>

        </td></tr>
      </table>

    </td></tr>
  </table>

</body>
</html>"""


def _build_diagnostics_section(report: dict) -> str:
    if not report:
        return ""

    status = report.get("status", "UNKNOWN")
    passed = report.get("passed", 0)
    total = report.get("total", 0)
    duration = report.get("duration_seconds", 0)
    failures = report.get("failures", [])

    status_colour = "#2e7d32" if status == "PASS" else "#c62828"
    status_bg = "#f1f8f1" if status == "PASS" else "#fff5f5"

    failure_rows = ""
    if failures:
        failure_rows = "".join(
            f"""<tr>
              <td style="padding:6px 0;font-family:Arial,Helvetica,sans-serif;font-size:12px;
                         color:#c62828;border-bottom:1px solid #f5f5f5;line-height:1.4">
                <b>{f['test'].split('.')[-1]}</b><br>
                <span style="color:#999">{f['message'][:120]}</span>
              </td>
            </tr>"""
            for f in failures
        )
        failure_table = f"""
        <table width="100%" cellpadding="0" cellspacing="0" style="margin-top:10px">
          {failure_rows}
        </table>"""
    else:
        failure_table = ""

    return f"""
    <table width="100%" cellpadding="0" cellspacing="0"
           style="background:{status_bg};border-radius:8px;margin-top:24px">
      <tr><td style="padding:16px 20px">
        <table width="100%" cellpadding="0" cellspacing="0">
          <tr>
            <td>
              <p style="font-family:Arial,Helvetica,sans-serif;font-size:13px;font-weight:700;
                         color:#888;text-transform:uppercase;letter-spacing:.6px;margin:0 0 4px 0">
                System Diagnostics
              </p>
              <p style="font-family:Arial,Helvetica,sans-serif;font-size:22px;font-weight:bold;
                         color:{status_colour};margin:0">{status}</p>
            </td>
            <td style="text-align:right">
              <p style="font-family:Arial,Helvetica,sans-serif;font-size:13px;color:#888;margin:0">
                {passed}/{total} tests passed
              </p>
              <p style="font-family:Arial,Helvetica,sans-serif;font-size:12px;color:#aaa;margin:4px 0 0">
                {duration:.1f}s
              </p>
            </td>
          </tr>
        </table>
        {failure_table}
      </td></tr>
    </table>"""


def _build_weekly_html(review: dict, name: str = "there", test_report: dict | None = None) -> str:
    week_summary = review.get("week_summary", "")
    what_worked = review.get("what_worked", [])
    what_didnt = review.get("what_didnt", [])
    lessons = review.get("lessons", [])
    applied = [c for c in review.get("applied_changes", []) if c["status"] in ("applied", "clamped")]
    rejected = [c for c in review.get("applied_changes", []) if c["status"] == "rejected"]

    def _bullets(items: list[str], colour: str = "#444") -> str:
        if not items:
            return '<p style="font-family:Arial,Helvetica,sans-serif;font-size:14px;color:#999;margin:0">None noted this week.</p>'
        return "".join(
            f'<p style="font-family:Arial,Helvetica,sans-serif;font-size:14px;color:{colour};margin:0 0 8px 0;padding-left:14px;border-left:3px solid #e0e0e0;line-height:1.5">{item}</p>'
            for item in items
        )

    def _section(title: str, content: str) -> str:
        return f"""
        <p style="font-family:Arial,Helvetica,sans-serif;font-size:13px;font-weight:700;color:#888;
                  text-transform:uppercase;letter-spacing:.6px;margin:24px 0 10px 0">{title}</p>
        {content}"""

    # Config changes table
    if applied:
        rows = "".join(
            f"""<tr>
              <td style="padding:8px 12px 8px 0;font-family:Arial,Helvetica,sans-serif;font-size:13px;
                         font-weight:600;color:#333;border-bottom:1px solid #f0f0f0;white-space:nowrap">{c['parameter']}</td>
              <td style="padding:8px 12px;font-family:Arial,Helvetica,sans-serif;font-size:13px;
                         color:#c62828;border-bottom:1px solid #f0f0f0">{c['old_value']}</td>
              <td style="padding:8px 12px;font-family:Arial,Helvetica,sans-serif;font-size:13px;
                         color:#2e7d32;font-weight:600;border-bottom:1px solid #f0f0f0">{c['new_value']}{"&nbsp;⚠︎" if c['status'] == 'clamped' else ""}</td>
              <td style="padding:8px 0;font-family:Arial,Helvetica,sans-serif;font-size:12px;
                         color:#666;border-bottom:1px solid #f0f0f0;line-height:1.4">{c.get('reason','')}</td>
            </tr>"""
            for c in applied
        )
        changes_block = f"""
        <table width="100%" cellpadding="0" cellspacing="0" style="margin-top:4px">
          <tr>
            <th style="text-align:left;padding:0 12px 8px 0;font-family:Arial,Helvetica,sans-serif;
                       font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.5px">Parameter</th>
            <th style="text-align:left;padding:0 12px 8px;font-family:Arial,Helvetica,sans-serif;
                       font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.5px">Before</th>
            <th style="text-align:left;padding:0 12px 8px;font-family:Arial,Helvetica,sans-serif;
                       font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.5px">After</th>
            <th style="text-align:left;padding:0 0 8px;font-family:Arial,Helvetica,sans-serif;
                       font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:.5px">Why</th>
          </tr>
          {rows}
        </table>
        {"<p style='font-family:Arial,Helvetica,sans-serif;font-size:11px;color:#aaa;margin:8px 0 0'>⚠︎ Value was clamped to the safety boundary.</p>" if any(c['status'] == 'clamped' for c in applied) else ""}
        <p style="font-family:Arial,Helvetica,sans-serif;font-size:12px;color:#888;margin:10px 0 0;line-height:1.5">
          These changes are live in <b>config.py</b> and will take effect from Monday&#39;s first run.
          Adjust manually if needed.
        </p>"""
    else:
        changes_block = '<p style="font-family:Arial,Helvetica,sans-serif;font-size:14px;color:#999;margin:0">No parameter changes this week — current settings are performing well.</p>'

    rejected_block = ""
    if rejected:
        rejected_block = _section(
            "Suggestions outside safe bounds (not applied)",
            "".join(
                f'<p style="font-family:Arial,Helvetica,sans-serif;font-size:13px;color:#aaa;margin:0 0 6px 0">'
                f'{c["parameter"]}: {c.get("reason","")} <i>(rejected: {c.get("rejection_reason","")})</i></p>'
                for c in rejected
            )
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>Weekly Trading Review</title>
</head>
<body style="margin:0;padding:0;background:#efefef;-webkit-text-size-adjust:100%">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#efefef">
    <tr><td align="center" style="padding:24px 12px">
      <table width="100%" cellpadding="0" cellspacing="0" style="max-width:600px;background:#ffffff;border-radius:10px;overflow:hidden">
        <tr><td style="padding:32px 28px 36px">

          <p style="font-family:Arial,Helvetica,sans-serif;font-size:16px;color:#555;margin:0 0 4px 0">Hi {name},</p>
          <p style="font-family:Arial,Helvetica,sans-serif;font-size:15px;color:#777;margin:0 0 24px 0">
            Here&#39;s your weekly self-review for the week ending <b>{date.today().isoformat()}</b>.
          </p>

          <!-- Summary banner -->
          <table width="100%" cellpadding="0" cellspacing="0" style="background:#f5f7ff;border-radius:10px;margin-bottom:8px">
            <tr><td style="padding:20px 24px">
              <p style="font-family:Arial,Helvetica,sans-serif;font-size:13px;font-weight:700;color:#888;
                         text-transform:uppercase;letter-spacing:.6px;margin:0 0 8px 0">How this week went</p>
              <p style="font-family:Arial,Helvetica,sans-serif;font-size:15px;color:#333;line-height:1.6;margin:0">{week_summary}</p>
            </td></tr>
          </table>

          {_section("What worked", _bullets(what_worked, "#2e7d32"))}
          {_section("What didn&#39;t work", _bullets(what_didnt, "#c62828"))}
          {_section("Lessons injected into next week&#39;s prompts", _bullets(lessons, "#1565c0"))}
          {_section("Config changes applied to config.py", changes_block)}
          {rejected_block}
          {_build_diagnostics_section(test_report)}

          <p style="font-family:Arial,Helvetica,sans-serif;font-size:11px;color:#bbb;margin:32px 0 0;text-align:center">
            Weekly self-review &nbsp;&#183;&nbsp; Your trading bot
          </p>

        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""


def send_weekly_review(review: dict, test_report: dict | None = None):
    applied_count = sum(1 for c in review.get("applied_changes", []) if c["status"] in ("applied", "clamped"))
    change_note = f" · {applied_count} config change{'s' if applied_count != 1 else ''} applied" if applied_count else ""
    diag_note = f" · tests {test_report.get('status', '')}" if test_report else ""
    _send_html(
        subject=f"Weekly Review {date.today().isoformat()}{change_note}{diag_note}",
        html_fn=lambda name: _build_weekly_html(review, name, test_report),
    )


def send_summary(record: dict):
    opening = record["account_before"]["portfolio_value"]
    pnl = record["daily_pnl"]
    ret_pct = (pnl / opening * 100) if opening else 0.0
    sign = "+" if pnl >= 0 else ""
    _send_html(
        subject=f"Trading Bot {record['date']} — {sign}${pnl:,.2f} ({sign}{ret_pct:.2f}%)",
        html_fn=lambda name: _build_html(record, name),
    )
