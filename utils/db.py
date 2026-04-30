"""
SQLite database layer. Single file at logs/investorbot.db.

All state that was previously scattered across JSON files is stored here:
  positions       — open position metadata (was positions_meta.json + fcntl)
  runs            — per-run records (was YYYY-MM-DD.json files)
  audit_events    — append-only event log (was audit.jsonl)
  decisions       — AI decision log (was decisions.jsonl)
  daily_baselines — open-of-day portfolio value (was daily_baseline.json)
  llm_usage       — token count and estimated cost per Claude call

WAL mode is enabled so readers never block writers.
All state mutations use explicit transactions to prevent partial writes.
"""

import json
import logging
import os
import sqlite3
from contextlib import contextmanager

from config import LOG_DIR

logger = logging.getLogger(__name__)

_DB_PATH = os.path.join(LOG_DIR, "investorbot.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    symbol       TEXT PRIMARY KEY,
    entry_date   TEXT NOT NULL,
    entry_price  REAL NOT NULL DEFAULT 0.0,
    signal       TEXT NOT NULL DEFAULT 'unknown',
    regime       TEXT NOT NULL DEFAULT 'UNKNOWN',
    confidence   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS runs (
    date                TEXT PRIMARY KEY,
    mode                TEXT NOT NULL DEFAULT 'open',
    run_id              TEXT,
    timestamp           TEXT NOT NULL,
    account_before      TEXT NOT NULL DEFAULT '{}',
    account_after       TEXT NOT NULL DEFAULT '{}',
    market_summary      TEXT NOT NULL DEFAULT '',
    position_decisions  TEXT NOT NULL DEFAULT '[]',
    buy_candidates      TEXT NOT NULL DEFAULT '[]',
    trades_executed     TEXT NOT NULL DEFAULT '[]',
    stop_losses         TEXT NOT NULL DEFAULT '[]',
    daily_pnl           REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS audit_events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id  TEXT,
    ts      TEXT NOT NULL,
    event   TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_audit_ts    ON audit_events(ts);
CREATE INDEX IF NOT EXISTS idx_audit_runid ON audit_events(run_id);

CREATE TABLE IF NOT EXISTS decisions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT,
    date           TEXT NOT NULL,
    mode           TEXT NOT NULL DEFAULT 'open',
    timestamp      TEXT NOT NULL,
    market_summary TEXT NOT NULL DEFAULT '',
    symbol         TEXT NOT NULL,
    action         TEXT NOT NULL,
    confidence     INTEGER,
    key_signal     TEXT,
    reasoning      TEXT,
    executed       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_decisions_date ON decisions(date);

CREATE TABLE IF NOT EXISTS daily_baselines (
    date            TEXT PRIMARY KEY,
    portfolio_value REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_usage (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT,
    ts            TEXT NOT NULL,
    model         TEXT NOT NULL DEFAULT '',
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd      REAL NOT NULL DEFAULT 0.0
);
"""


def _connect() -> sqlite3.Connection:
    os.makedirs(LOG_DIR, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db():
    """Context manager yielding an open connection. Commits on clean exit, rolls back on error."""
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


_initialized = False


def init_db():
    """Create schema (idempotent) and migrate any existing JSON-file state."""
    global _initialized
    if _initialized:
        return
    with get_db() as conn:
        conn.executescript(_SCHEMA)
    _migrate_json_state()
    _initialized = True
    logger.debug(f"Database ready: {_DB_PATH}")


# ── Migration from legacy JSON files ─────────────────────────────────────────

def _migrate_json_state():
    """One-time import of existing JSON files into SQLite. Safe to call repeatedly."""
    _migrate_positions()
    _migrate_runs()
    _migrate_audit()
    _migrate_decisions()
    _migrate_baseline()


def _migrate_positions():
    meta_path = os.path.join(LOG_DIR, "positions_meta.json")
    if not os.path.exists(meta_path):
        return
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        with get_db() as conn:
            existing = {row[0] for row in conn.execute("SELECT symbol FROM positions")}
            for sym, data in meta.items():
                if sym not in existing:
                    conn.execute(
                        "INSERT OR IGNORE INTO positions VALUES (?,?,?,?,?,?)",
                        (sym, data.get("entry_date", ""), data.get("entry_price", 0.0),
                         data.get("signal", "unknown"), data.get("regime", "UNKNOWN"),
                         data.get("confidence", 0)),
                    )
        logger.info(f"Migrated {len(meta)} position(s) from positions_meta.json")
    except Exception as e:
        logger.warning(f"Position migration skipped: {e}")


def _migrate_runs():
    import re
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}(-midday|-close)?\.json$")
    non_run = {"positions_meta.json", "signal_stats.json", "daily_baseline.json",
               "backtest_results.json"}
    try:
        files = [f for f in os.listdir(LOG_DIR)
                 if f.endswith(".json") and f not in non_run and pattern.match(f)]
    except FileNotFoundError:
        return
    migrated = 0
    for fname in sorted(files):
        try:
            with open(os.path.join(LOG_DIR, fname)) as f:
                r = json.load(f)
            if "date" not in r or "account_after" not in r:
                continue
            with get_db() as conn:
                exists = conn.execute(
                    "SELECT 1 FROM runs WHERE date=?", (r["date"],)
                ).fetchone()
                if not exists:
                    conn.execute(
                        "INSERT OR IGNORE INTO runs "
                        "(date,mode,run_id,timestamp,account_before,account_after,"
                        "market_summary,position_decisions,buy_candidates,"
                        "trades_executed,stop_losses,daily_pnl) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (r["date"],
                         "open" if "-" not in r["date"].lstrip("0123456789-")[:1] else r["date"].split("-")[-1],
                         r.get("run_id"),
                         r.get("timestamp", ""),
                         json.dumps(r.get("account_before", {})),
                         json.dumps(r.get("account_after", {})),
                         r.get("market_summary", ""),
                         json.dumps(r.get("position_decisions", [])),
                         json.dumps(r.get("buy_candidates", [])),
                         json.dumps(r.get("trades_executed", [])),
                         json.dumps(r.get("stop_losses_triggered", [])),
                         r.get("daily_pnl", 0.0)),
                    )
                    migrated += 1
        except Exception as e:
            logger.warning(f"Run migration skipped for {fname}: {e}")
    if migrated:
        logger.info(f"Migrated {migrated} run record(s) from JSON files")


def _migrate_audit():
    audit_path = os.path.join(LOG_DIR, "audit.jsonl")
    if not os.path.exists(audit_path):
        return
    try:
        with get_db() as conn:
            count = conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
            if count > 0:
                return  # already migrated
            rows = []
            with open(audit_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ev = json.loads(line)
                    payload = {k: v for k, v in ev.items() if k not in ("ts", "event")}
                    rows.append((ev.get("run_id"), ev.get("ts", ""), ev.get("event", ""),
                                 json.dumps(payload)))
            conn.executemany(
                "INSERT INTO audit_events (run_id,ts,event,payload) VALUES (?,?,?,?)", rows
            )
        logger.info(f"Migrated {len(rows)} audit event(s) from audit.jsonl")
    except Exception as e:
        logger.warning(f"Audit migration skipped: {e}")


def _migrate_decisions():
    dec_path = os.path.join(LOG_DIR, "decisions.jsonl")
    if not os.path.exists(dec_path):
        return
    try:
        with get_db() as conn:
            count = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
            if count > 0:
                return
            rows = []
            with open(dec_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    rows.append((d.get("run_id"), d.get("date", ""), d.get("mode", "open"),
                                 d.get("ts", ""), d.get("market_summary", ""),
                                 d.get("symbol", ""), d.get("action", ""),
                                 d.get("confidence"), d.get("key_signal"),
                                 d.get("reasoning"), int(d.get("executed", False))))
            conn.executemany(
                "INSERT INTO decisions (run_id,date,mode,timestamp,market_summary,"
                "symbol,action,confidence,key_signal,reasoning,executed) VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?)", rows
            )
        logger.info(f"Migrated {len(rows)} decision(s) from decisions.jsonl")
    except Exception as e:
        logger.warning(f"Decision migration skipped: {e}")


def _migrate_baseline():
    bl_path = os.path.join(LOG_DIR, "daily_baseline.json")
    if not os.path.exists(bl_path):
        return
    try:
        with open(bl_path) as f:
            data = json.load(f)
        if "date" in data and "portfolio_value" in data:
            with get_db() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO daily_baselines VALUES (?,?)",
                    (data["date"], data["portfolio_value"]),
                )
    except Exception as e:
        logger.warning(f"Baseline migration skipped: {e}")
