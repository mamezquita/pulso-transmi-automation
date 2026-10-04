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


H = lambda n: NOW - timedelta(hours=n)
CLEAN = [{'quarantine': []}] * 4


def test_healthy_data_has_no_alerts():
    assert wd.health(NOW, H(0), H(0), H(1.5), CLEAN, []) == {}


def test_observations_behind_the_cycle_cutoff_alert():
    # 3-oct: el stream cambió de formato y lo guardado quedó 4 h detrás del corte.
    alerts = wd.health(NOW, H(0), H(4), H(4), CLEAN, [])
    assert set(alerts) == {'datos', 'modelo'}
    assert 'Datos atrasados' == alerts['datos'][0] and '4.0 h' in alerts['datos'][1]


def test_professor_pause_is_not_an_alert():
    # Sin ciclos nuevos el corte no avanza: datos y modelo siguen al día respecto a él.
    assert wd.health(NOW, H(18), H(18), H(19), CLEAN, []) == {}


def test_quarantine_alert_needs_four_attempts_and_ignores_declared_missing_values():
    bad = {'quarantine': [{'reason': 'Falta el campo demand'}]}
    missing = {'quarantine': [{'reason': wd.MISSING_REASON}]}
    assert 'cuarentena' in wd.health(NOW, H(0), H(0), H(1), [bad] * 4, [])
    assert 'cuarentena' not in wd.health(NOW, H(0), H(0), H(1), [bad] * 3 + [{'quarantine': []}], [])
    assert 'cuarentena' not in wd.health(NOW, H(0), H(0), H(1), [missing] * 4, [])


def test_recent_format_change_opens_its_own_alert():
    ch = {'formato_id': 7, 'fuente': 'stream', 'primera_vez': H(2).isoformat(),
          'cambios': {'data[].measurement.value': {'antes': None, 'ahora': 'str:999.99'}}}
    old = {**ch, 'formato_id': 3, 'primera_vez': H(30).isoformat()}
    alerts = wd.health(NOW, H(0), H(0), H(1), CLEAN, [ch, old])
    assert list(alerts) == ['formato-7'] and 'measurement.value' in alerts['formato-7'][1]


def test_issues_are_opened_once_and_closed_when_resolved_except_format():
    calls, issues = [], [{'title': '[Alerta] Modelo atrasado', 'number': 5},
                         {'title': '[Alerta] Cambio de formato en stream (#7)', 'number': 6},
                         {'title': 'otra cosa', 'number': 7}]
    def api(method, path, body=None):
        calls.append((method, path, body and body.get('title', body.get('state'))))
        return issues if method == 'GET' else {}
    opened, closed = wd.sync_issues({'datos': ('Datos atrasados', 'x')}, api=api)
    assert opened == ['[Alerta] Datos atrasados'] and closed == ['[Alerta] Modelo atrasado']
    assert ('PATCH', '/issues/5', 'closed') in calls and not any(p == '/issues/6' for _, p, _ in calls)
    opened, _ = wd.sync_issues({'datos': ('Datos atrasados', 'x')},
                               api=lambda m, p, b=None: [{'title': '[Alerta] Datos atrasados', 'number': 9}] if m == 'GET' else {})
    assert opened == []  # ya abierto: no se repite
