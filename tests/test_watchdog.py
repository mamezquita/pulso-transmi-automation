from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('watchdog', Path(__file__).parents[1]/'scripts/watchdog.py')
wd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wd)

NOW = datetime(2026, 9, 27, 15, 20, tzinfo=timezone.utc)
UNTIL = datetime(2026, 10, 10, 12, 59, tzinfo=timezone.utc)
M = lambda n: NOW - timedelta(minutes=n)


def test_healthy():
    assert wd.diagnose(NOW, UNTIL, M(1), M(44)) == 'ok'


def test_dead_session_chain_is_relaunched():
    assert wd.diagnose(NOW, UNTIL, M(25), M(44)) == 'stalled'
    assert wd.diagnose(NOW, UNTIL, None, None) == 'stalled'


def test_missed_window_alerts():
    assert wd.diagnose(NOW, UNTIL, M(1), M(104)) == 'no_recent_delivery'
    assert wd.diagnose(NOW, UNTIL, M(1), None) == 'no_recent_delivery'


def test_campaign_over_is_quiet():
    assert wd.diagnose(UNTIL, UNTIL, None, None) == 'campaign_over'


def test_receipt_parsing_ignores_rows_without_receipt():
    rows = [{'respuesta': None}, {'respuesta': {'response': {'error': 'x'}}},
            {'respuesta': {'response': {'submission_id': 's', 'received_at': '2026-09-26T12:55:10Z'}}}]
    assert wd.latest_receipt(rows) == datetime(2026, 9, 26, 12, 55, 10, tzinfo=timezone.utc)
