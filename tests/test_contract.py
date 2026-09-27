"""Trampas de formato que el profesor puede introducir."""
import json
from pathlib import Path

import httpx
import pandas as pd
import pytest

from pulso_transmi.collector import collect
from pulso_transmi.contract import (ContractError, normalize_cycle_contract,
                                    normalize_observations, parse_instants)
from pulso_transmi import automation

REAL = json.loads((Path(__file__).parent / 'fixtures/contrato_real.json').read_text())
UTC = lambda s: pd.Timestamp(s, tz='UTC')


def row(**kw):
    base = {'station_id': '02300', 'observed_at': '2026-09-14T15:00:00Z',
            'demand': 56, 'released_at': '2026-09-27T01:45:28.272422Z'}
    return {**base, **kw}


# --- El formato actual no cambia nada -------------------------------------------------

def test_current_contract_is_unchanged():
    assert normalize_cycle_contract(REAL) == REAL


def test_current_observation_is_unchanged():
    good, bad, _ = normalize_observations([row()])
    assert bad == [] and good == [row()]


# --- Fechas ---------------------------------------------------------------------------

@pytest.mark.parametrize('value', [
    '2026-09-14T15:00:00Z', '2026-09-14T10:00:00-05:00', '2026/09/14 15:00:00',
    '14/09/2026 15:00', '09/14/2026 15:00:00Z', '14-09-2026T15:00:00Z',
    '2026.09.14 15:00 UTC', 1789398000, 1789398000000,
])
def test_unambiguous_date_formats(value):
    assert parse_instants([value]) == [UTC('2026-09-14 15:00')]


def test_day_month_ambiguity_is_resolved_by_the_rest_of_the_page():
    # 05/09 solo es ambiguo; 13/09 en la misma página fija día/mes.
    got = parse_instants(['05/09/2026 10:00', '13/09/2026 10:00'])
    assert got == [UTC('2026-09-05 10:00'), UTC('2026-09-13 10:00')]


def test_day_month_ambiguity_is_resolved_by_reference():
    got = parse_instants(['05/09/2026 10:00'], near=UTC('2026-09-06'))
    assert got == [UTC('2026-09-05 10:00')]
    got = parse_instants(['05/09/2026 10:00'], near=UTC('2026-05-08'))
    assert got == [UTC('2026-05-09 10:00')]


def test_ambiguous_without_reference_is_rejected_not_guessed():
    with pytest.raises(ContractError):
        parse_instants(['05/09/2026 10:00'])


def test_year_first_defaults_to_year_month_day():
    assert parse_instants(['2026/09/05 10:00']) == [UTC('2026-09-05 10:00')]


@pytest.mark.parametrize('value', ['ayer', '2026-13-45', '31/31/2026', True, None])
def test_garbage_dates_rejected(value):
    with pytest.raises(ContractError):
        parse_instants([value])


# --- Tipos y nombres de campos --------------------------------------------------------

def test_types_and_aliases_are_translated():
    good, bad, fp = normalize_observations([
        {'estacion_id': 2300, 'fecha': '14/09/2026 15:00', 'demanda': '56',
         'publicado_en': '2026-09-27T01:45:28.272422Z'},
        row(station_id='2300', demand=56.0, observed_at='2026-09-14T15:15:00Z'),
    ])
    assert bad == []
    assert good[0] == row() and good[1]['observed_at'] == '2026-09-14T15:15:00Z'
    assert fp['demand'] == ['demand:float', 'demanda:str-num']


@pytest.mark.parametrize('bad_row', [
    row(demand=12.5), row(demand=-1), row(demand=True), row(demand='doce'),
    row(station_id='ABC'), row(station_id=123456), row(observed_at='2026-09-14T15:07:00Z'),
    {'station_id': '02300'}, 'no es un objeto',
])
def test_invalid_rows_go_to_quarantine_without_stopping_the_page(bad_row):
    good, bad, _ = normalize_observations([row(), bad_row])
    assert good == [row()]
    assert len(bad) == 1 and bad[0]['row'] == bad_row and bad[0]['reason']


# --- Contrato del ciclo ---------------------------------------------------------------

def test_contract_with_new_date_format_and_types_is_canonicalized():
    changed = json.loads(json.dumps(REAL))
    to_dmy = lambda s: pd.Timestamp(s).strftime('%d/%m/%Y %H:%M')
    for k in ('data_cutoff', 'origin_at', 'forecast_start_at', 'forecast_end_at'):
        changed[k] = to_dmy(changed[k])
    for t in changed['targets']:
        t['target_at'] = to_dmy(t['target_at'])
        t['station_id'] = int(t['station_id'])
        t['horizon_minutes'] = str(t['horizon_minutes'])
    changed['expected_predictions'] = '48'
    assert normalize_cycle_contract(changed) == REAL


def test_inconsistent_contract_is_rejected():
    changed = json.loads(json.dumps(REAL))
    changed['targets'][0]['target_at'] = '2026-09-14T17:15:00Z'
    with pytest.raises(ContractError):
        normalize_cycle_contract(changed)


# --- Colector y aislamiento -----------------------------------------------------------

class State:
    def __init__(self):
        self.data = {'cursor': None, 'revision': 0}; self.saved = []
    def state(self): return dict(self.data)
    def latest_observation(self, cutoff): return '2026-09-14T14:45:00Z'
    def save_page(self, rev, rows, cursor):
        self.saved += rows; self.data = {'revision': rev + 1, 'cursor': cursor}
        return {**self.data, 'inserted': len(rows)}


def api(rows, cursor=None):
    body = {'data': rows, 'next_cursor': cursor}
    return httpx.Client(base_url='https://example.test',
                        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)))


def test_collector_quarantines_bad_rows_and_saves_good_ones():
    store = State()
    with api([row(), row(demand='x')]) as client:
        out = collect(store, client)
    assert out['inserted'] == 1 and out['quarantined'] == 1
    assert out['quarantine'][0]['row']['demand'] == 'x'
    assert store.saved == [row()]


def test_fully_unreadable_page_does_not_advance_cursor():
    store = State()
    with api([row(observed_at='ayer')], cursor='next') as client:
        with pytest.raises(ContractError):
            collect(store, client)
    assert store.data['cursor'] is None and store.saved == []


class AttemptStore:
    def start_attempt(self, *a): pass
    def finish_attempt(self, attempt_id, result): self.finished = result
    def observe_cycle(self, cycle): pass
    def evaluate_deliveries(self): return {}


def test_collector_failure_never_blocks_submission(monkeypatch):
    def broken(*a): raise ContractError('Página completa en cuarentena: Formato de fecha no reconocido')
    monkeypatch.setattr(automation, 'collect', broken)
    monkeypatch.setattr(automation, 'check_current_cycle', lambda *a: {'cycle_id': 'cyc_x'})
    monkeypatch.setattr(automation, 'forecast', lambda *a, **k: {'status': 'accepted', 'submitted': True})
    store = AttemptStore()
    result = automation.execute(store, None)
    assert result['forecast']['submitted'] and result['degraded']
    assert 'status' not in result
    assert result['collector'] == {'status': 'error', 'error_type': 'ContractError',
                                   'detail': 'Página completa en cuarentena: Formato de fecha no reconocido'}
