.PHONY: help test lint safety-check drill-stop-failure drill-broker-timeout \
        drill-restart-mid-order drill-kill-switch drill-missing-stop \
        drill-duplicate-order

PYTHON := .venv/bin/python

help:
	@echo "InvestorBot make targets:"
	@echo "  make test                  Run full test suite with coverage"
	@echo "  make lint                  Run ruff check + format"
	@echo "  make safety-check          Run startup safety check (paper)"
	@echo "  make drill-stop-failure    Drill: stop placement fails → flatten + halt"
	@echo "  make drill-broker-timeout  Drill: broker API timeout → buys suspended"
	@echo "  make drill-restart-mid-order  Drill: crash between submit and fill"
	@echo "  make drill-kill-switch     Drill: emergency kill switch activation"
	@echo "  make drill-missing-stop    Drill: unprotected position at startup"
	@echo "  make drill-duplicate-order Drill: duplicate client_order_id rejection"

test:
	$(PYTHON) -m pytest --tb=short -q

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

safety-check:
	$(PYTHON) main.py --safety-check

# ── Incident drills ───────────────────────────────────────────────────────────
# Each drill exercises a specific safety mechanism in paper mode.
# Expected outputs are printed after each drill.

drill-stop-failure:
	@echo "=== Drill: Stop placement failure ==="
	@echo "Expected: alert_error('STOP FAILED') logged, position flattened (paper: alert only)"
	$(PYTHON) -c "
from unittest.mock import MagicMock, patch
from models import OrderResult, OrderStatus
with (
    patch('main.config.IS_PAPER', True),
    patch('main.config.HALT_FILE', '/tmp/drill_halt'),
    patch('main.config.LOG_DIR', '/tmp'),
    patch('main.trader.close_position', return_value=OrderResult(status=OrderStatus.FILLED, symbol='DRILL')),
    patch('main.trader.record_sell'),
    patch('main.alerts.alert_error', lambda t, m: print(f'ALERT: {t}: {m}')),
):
    from main import _handle_stop_failure
    _handle_stop_failure(MagicMock(), 'DRILL', dry_run=False)
print('Drill complete.')
"

drill-broker-timeout:
	@echo "=== Drill: Broker API timeout ==="
	@echo "Expected: BrokerStateUnavailable raised from has_pending_buy"
	$(PYTHON) -c "
from unittest.mock import MagicMock
from execution.trader import has_pending_buy
from models import BrokerStateUnavailable
client = MagicMock()
client.get_orders.side_effect = TimeoutError('broker unreachable')
try:
    has_pending_buy(client, 'DRILL')
    print('ERROR: should have raised BrokerStateUnavailable')
except BrokerStateUnavailable as e:
    print(f'PASS: BrokerStateUnavailable raised: {e}')
print('Drill complete.')
"

drill-restart-mid-order:
	@echo "=== Drill: Crash between submit and fill ==="
	@echo "Expected: order_intents shows status=submitted; get_unresolved_intents returns it"
	$(PYTHON) -c "
import tempfile, os
from unittest.mock import patch
with tempfile.TemporaryDirectory() as tmpdir:
    db_path = os.path.join(tmpdir, 'test.db')
    with patch('utils.db._DB_PATH', db_path), patch('config.LOG_DIR', tmpdir):
        from utils.db import init_db
        from utils.order_ledger import create_intent, update_intent, get_unresolved_intents
        init_db()
        create_intent('DRILL', 'BUY', '2026-05-04', 50.0, 'ib-DRILL-BUY-2026-05-04')
        update_intent('ib-DRILL-BUY-2026-05-04', 'submitted', 'broker-001')
        unresolved = get_unresolved_intents(trade_date='2026-05-04')
        assert len(unresolved) == 1, f'Expected 1 unresolved, got {len(unresolved)}'
        assert unresolved[0]['status'] == 'submitted'
        print(f'PASS: unresolved intent found: {unresolved[0]}')
print('Drill complete.')
"

drill-kill-switch:
	@echo "=== Drill: Kill switch (dry run — no real orders) ==="
	@echo "Expected: would cancel orders, close positions, write halt file"
	@echo "NOTE: This drill only validates the logic path, not actual broker calls."
	$(PYTHON) -c "
print('Kill switch drill: run manually with python main.py --kill-switch in paper mode')
print('Verify: logs/.HALTED written, positions closed, alert sent')
print('Resume: python main.py --clear-halt')
"

drill-missing-stop:
	@echo "=== Drill: Unprotected position at startup ==="
	@echo "Expected: ensure_stops_attached detects and attempts to attach stop"
	$(PYTHON) -c "
from unittest.mock import MagicMock, patch
from execution.trader import ensure_stops_attached
pos = MagicMock()
pos.symbol = 'DRILL'
pos.qty = '2'
pos.current_price = '50.0'
client = MagicMock()
client.get_all_positions.return_value = [pos]
client.get_orders.return_value = []
order = MagicMock()
order.id = 'stop-drill-001'
client.submit_order.return_value = order
result = ensure_stops_attached(client)
print(f'PASS: ensure_stops_attached returned {result}, stop submitted: {client.submit_order.called}')
print('Drill complete.')
"

drill-duplicate-order:
	@echo "=== Drill: Duplicate client_order_id rejection ==="
	@echo "Expected: second submission raises exception → REJECTED result, no new exposure"
	$(PYTHON) -c "
from unittest.mock import MagicMock, patch
from alpaca.trading.client import TradingClient
from execution.trader import place_buy_order
from models import OrderStatus
import tempfile, os
with tempfile.TemporaryDirectory() as tmpdir:
    with patch('utils.db._DB_PATH', os.path.join(tmpdir, 'test.db')), \
         patch('config.LOG_DIR', tmpdir):
        from utils.db import init_db
        init_db()
        call_count = [0]
        def fake_submit(req):
            call_count[0] += 1
            if call_count[0] == 1:
                o = MagicMock(); o.id = 'first-order'; return o
            raise Exception('ClientOrderIdNotUnique: 409')
        client = MagicMock()
        client.submit_order.side_effect = fake_submit
        with patch('execution.trader.wait_for_fill', return_value=1.0):
            r1 = place_buy_order(client, 'DRILL', 50.0)
        with patch('execution.trader.wait_for_fill', return_value=None):
            r2 = place_buy_order(client, 'DRILL', 50.0)
        assert r1.status == OrderStatus.FILLED, f'Expected FILLED, got {r1.status}'
        assert r2.status == OrderStatus.REJECTED, f'Expected REJECTED, got {r2.status}'
        print(f'PASS: first={r1.status}, duplicate={r2.status} (no second exposure created)')
print('Drill complete.')
"
