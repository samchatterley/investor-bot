import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import utils.db as db_mod
from utils.db import _SCHEMA


class DbTestBase(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self._db_patcher = patch("utils.db._DB_PATH", self.db_path)
        self._log_patcher = patch("utils.db.LOG_DIR", self.tmpdir)
        self._db_patcher.start()
        self._log_patcher.start()
        # Create the schema so tables exist for migration calls
        from utils.db import get_db
        with get_db() as conn:
            conn.executescript(_SCHEMA)

    def tearDown(self):
        self._db_patcher.stop()
        self._log_patcher.stop()
        shutil.rmtree(self.tmpdir)
        # Reset the singleton flag so other test modules aren't affected
        db_mod._initialized = False


class TestMigratePositions(DbTestBase):

    def test_no_op_when_file_missing(self):
        from utils.db import _migrate_positions
        _migrate_positions()  # must not raise

    def test_inserts_from_positions_meta(self):
        meta = {
            "AAPL": {"entry_date": "2026-04-01", "entry_price": 180.0,
                     "signal": "momentum", "regime": "BULL", "confidence": 8},
        }
        with open(os.path.join(self.tmpdir, "positions_meta.json"), "w") as f:
            json.dump(meta, f)
        from utils.db import _migrate_positions, get_db
        _migrate_positions()
        with get_db() as conn:
            rows = conn.execute("SELECT symbol FROM positions").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "AAPL")

    def test_skips_symbols_already_in_db(self):
        meta = {"AAPL": {"entry_date": "2026-04-01", "entry_price": 180.0}}
        with open(os.path.join(self.tmpdir, "positions_meta.json"), "w") as f:
            json.dump(meta, f)
        from utils.db import _migrate_positions, get_db
        with get_db() as conn:
            conn.execute(
                "INSERT INTO positions (symbol, entry_date, entry_price, signal, regime, confidence) "
                "VALUES (?,?,?,?,?,?)",
                ("AAPL", "2026-04-01", 180.0, "momentum", "BULL", 8),
            )
        _migrate_positions()
        with get_db() as conn:
            count = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
        self.assertEqual(count, 1)  # not duplicated

    def test_handles_corrupt_json_gracefully(self):
        with open(os.path.join(self.tmpdir, "positions_meta.json"), "w") as f:
            f.write("not json {{{")
        from utils.db import _migrate_positions
        _migrate_positions()  # must not raise


class TestMigrateRuns(DbTestBase):

    def test_no_op_when_log_dir_missing(self):
        from utils.db import _migrate_runs
        with patch("utils.db.LOG_DIR", "/nonexistent/__test_path__"):
            _migrate_runs()  # must not raise

    def test_inserts_run_from_json_file(self):
        run_data = {
            "date": "2026-04-01",
            "account_before": {"portfolio_value": 100_000},
            "account_after": {"portfolio_value": 101_000},
            "daily_pnl": 1000.0,
            "timestamp": "2026-04-01T09:30:00Z",
        }
        with open(os.path.join(self.tmpdir, "2026-04-01.json"), "w") as f:
            json.dump(run_data, f)
        from utils.db import _migrate_runs, get_db
        _migrate_runs()
        with get_db() as conn:
            row = conn.execute("SELECT date FROM runs WHERE date=?", ("2026-04-01",)).fetchone()
        self.assertIsNotNone(row)

    def test_skips_files_missing_required_fields(self):
        with open(os.path.join(self.tmpdir, "2026-04-01.json"), "w") as f:
            json.dump({"date": "2026-04-01"}, f)  # missing account_after
        from utils.db import _migrate_runs, get_db
        _migrate_runs()
        with get_db() as conn:
            row = conn.execute("SELECT date FROM runs WHERE date=?", ("2026-04-01",)).fetchone()
        self.assertIsNone(row)

    def test_skips_non_run_files(self):
        with open(os.path.join(self.tmpdir, "positions_meta.json"), "w") as f:
            json.dump({"AAPL": {}}, f)
        from utils.db import _migrate_runs, get_db
        _migrate_runs()
        with get_db() as conn:
            count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        self.assertEqual(count, 0)

    def test_does_not_duplicate_existing_record(self):
        run_data = {
            "date": "2026-04-01",
            "account_before": {},
            "account_after": {"portfolio_value": 101_000},
            "timestamp": "",
        }
        with open(os.path.join(self.tmpdir, "2026-04-01.json"), "w") as f:
            json.dump(run_data, f)
        from utils.db import _migrate_runs, get_db
        with get_db() as conn:
            conn.execute(
                "INSERT INTO runs (date,mode,timestamp,account_before,account_after,"
                "market_summary,position_decisions,buy_candidates,trades_executed,"
                "stop_losses,daily_pnl) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("2026-04-01", "open", "", "{}", "{}", "", "[]", "[]", "[]", "[]", 0.0),
            )
        _migrate_runs()
        with get_db() as conn:
            count = conn.execute("SELECT COUNT(*) FROM runs WHERE date=?", ("2026-04-01",)).fetchone()[0]
        self.assertEqual(count, 1)


class TestMigrateAudit(DbTestBase):

    def test_no_op_when_file_missing(self):
        from utils.db import _migrate_audit
        _migrate_audit()  # must not raise

    def test_inserts_events_from_jsonl(self):
        events = [
            {"ts": "2026-04-01T09:30:00Z", "event": "RUN_START", "mode": "open"},
            {"ts": "2026-04-01T09:35:00Z", "event": "ORDER_PLACED", "symbol": "AAPL"},
        ]
        audit_path = os.path.join(self.tmpdir, "audit.jsonl")
        with open(audit_path, "w") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")
        from utils.db import _migrate_audit, get_db
        _migrate_audit()
        with get_db() as conn:
            count = conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
        self.assertEqual(count, 2)

    def test_skips_when_events_already_present(self):
        audit_path = os.path.join(self.tmpdir, "audit.jsonl")
        with open(audit_path, "w") as f:
            f.write(json.dumps({"ts": "2026-04-01T09:30:00Z", "event": "RUN_START"}) + "\n")
        from utils.db import _migrate_audit, get_db
        with get_db() as conn:
            conn.execute(
                "INSERT INTO audit_events (ts,event,payload) VALUES (?,?,?)",
                ("2026-04-01T09:00:00Z", "EXISTING", "{}"),
            )
        _migrate_audit()
        with get_db() as conn:
            count = conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
        self.assertEqual(count, 1)  # file events not added when table is non-empty

    def test_skips_empty_lines(self):
        audit_path = os.path.join(self.tmpdir, "audit.jsonl")
        with open(audit_path, "w") as f:
            f.write("\n")
            f.write(json.dumps({"ts": "2026-04-01T09:30:00Z", "event": "RUN_START"}) + "\n")
            f.write("\n")
        from utils.db import _migrate_audit, get_db
        _migrate_audit()
        with get_db() as conn:
            count = conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
        self.assertEqual(count, 1)


class TestMigrateDecisions(DbTestBase):

    def test_no_op_when_file_missing(self):
        from utils.db import _migrate_decisions
        _migrate_decisions()  # must not raise

    def test_inserts_decisions_from_jsonl(self):
        decision = {
            "run_id": "r1", "date": "2026-04-01", "mode": "open",
            "ts": "2026-04-01T09:30:00Z", "market_summary": "",
            "symbol": "AAPL", "action": "BUY", "confidence": 8, "executed": True,
        }
        dec_path = os.path.join(self.tmpdir, "decisions.jsonl")
        with open(dec_path, "w") as f:
            f.write(json.dumps(decision) + "\n")
        from utils.db import _migrate_decisions, get_db
        _migrate_decisions()
        with get_db() as conn:
            count = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        self.assertEqual(count, 1)

    def test_skips_when_decisions_already_present(self):
        dec_path = os.path.join(self.tmpdir, "decisions.jsonl")
        with open(dec_path, "w") as f:
            f.write(json.dumps({"symbol": "AAPL", "action": "BUY",
                                 "date": "2026-04-01", "mode": "open", "ts": ""}) + "\n")
        from utils.db import _migrate_decisions, get_db
        with get_db() as conn:
            conn.execute(
                "INSERT INTO decisions (date,mode,timestamp,market_summary,symbol,action,executed)"
                " VALUES (?,?,?,?,?,?,?)",
                ("2026-04-01", "open", "", "", "MSFT", "HOLD", 0),
            )
        _migrate_decisions()
        with get_db() as conn:
            count = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        self.assertEqual(count, 1)  # file events not added when table is non-empty


class TestMigrateBaseline(DbTestBase):

    def test_no_op_when_file_missing(self):
        from utils.db import _migrate_baseline
        _migrate_baseline()  # must not raise

    def test_inserts_baseline_from_json(self):
        bl_path = os.path.join(self.tmpdir, "daily_baseline.json")
        with open(bl_path, "w") as f:
            json.dump({"date": "2026-04-01", "portfolio_value": 100_000.0}, f)
        from utils.db import _migrate_baseline, get_db
        _migrate_baseline()
        with get_db() as conn:
            row = conn.execute(
                "SELECT portfolio_value FROM daily_baselines WHERE date=?", ("2026-04-01",)
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertAlmostEqual(row[0], 100_000.0)

    def test_skips_when_required_keys_absent(self):
        bl_path = os.path.join(self.tmpdir, "daily_baseline.json")
        with open(bl_path, "w") as f:
            json.dump({"something_else": "value"}, f)  # no date / portfolio_value
        from utils.db import _migrate_baseline, get_db
        _migrate_baseline()  # must not raise
        with get_db() as conn:
            count = conn.execute("SELECT COUNT(*) FROM daily_baselines").fetchone()[0]
        self.assertEqual(count, 0)

    def test_handles_corrupt_json_gracefully(self):
        bl_path = os.path.join(self.tmpdir, "daily_baseline.json")
        with open(bl_path, "w") as f:
            f.write("not json")
        from utils.db import _migrate_baseline
        _migrate_baseline()  # must not raise


class TestInitDb(DbTestBase):

    def test_tables_exist_after_init(self):
        db_mod._initialized = False
        from utils.db import init_db, get_db
        init_db()
        with get_db() as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        for expected in ("positions", "runs", "audit_events", "decisions",
                         "daily_baselines", "llm_usage"):
            self.assertIn(expected, tables)

    def test_second_call_is_no_op(self):
        db_mod._initialized = False
        from utils.db import init_db
        init_db()
        init_db()  # second call — must not raise or reset state
        self.assertTrue(db_mod._initialized)
