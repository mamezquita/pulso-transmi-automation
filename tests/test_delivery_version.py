import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pulso_transmi import automation
from pulso_transmi.forecaster import prepare

CUTOFF = '2026-09-14T15:00:00Z'
TARGETS = [{'station_id': s, 'target_at': t, 'horizon_minutes': h}
           for s in ('02300', '03000') for t, h in (('2026-09-14T15:15:00Z', 15), ('2026-09-14T16:00:00Z', 60))]
CYCLE = {'cycle_id': 'cyc_test', 'state': 'open', 'data_cutoff': CUTOFF, 'closes_at': '2099-01-01T00:00:00Z',
         'expected_predictions': 4, 'targets': TARGETS}
V1 = {'version': '1.0', 'metodo': 'modelo_base', 'parametros': {}, 'estado': 'sombra'}
V2 = {'version': '2.0', 'metodo': 'ratio_ultima_hora', 'estado': 'sombra',
      'parametros': {'ventana_obs': 4, 'limite_min': 0.5, 'limite_max': 2.0}}


def last_hour(a=150, b=100):
    times = pd.date_range('2026-09-14T14:15:00Z', CUTOFF, freq='15min')
    return [{'station_id': s, 'observed_at': t.isoformat(), 'demand': v}
            for t in times for s, v in (('02300', a), ('03000', b))]


class Store:
    def __init__(self, selected=V1, existing=None, fail_record=False):
        self.selected = selected; self.record = existing; self.recorded = []; self.fail_record = fail_record
        self.meta = {'sha256': 'a' * 64, 'version': 'calendar-test',
                     'metadata': {'training_data_end': '2026-09-10T00:00:00Z', 'trained_at': '2026-09-23T00:00:00Z'}}
    def job(self, c): return copy.deepcopy(self.record) if self.record and self.record.get('respuesta') else None
    def active_model(self): return self.meta
    def latest_observation(self, c): return CUTOFF
    def load_model(self, r):
        return SimpleNamespace(end=r['metadata']['training_data_end'],
                               predict=lambda d: np.full(len(d), 100.4))
    def delivery_version(self):
        if isinstance(self.selected, Exception): raise self.selected
        return self.selected
    def observations_window(self, start, end): return last_hour()
    def reserve(self, payload, sha):
        if self.record is None:
            self.record = {'ciclo_id': payload['cycle_id'], 'client_run_id': payload['client_run_id'],
                           'payload': copy.deepcopy(payload), 'respuesta': None}
        return copy.deepcopy(self.record)
    def record_delivery_version(self, cycle_id, selection):
        if self.fail_record: raise RuntimeError('supabase caído')
        self.recorded.append((cycle_id, selection))


def values(job):
    return {(p['station_id'], p['target_at']): p['value'] for p in job['payload']['predictions']}


def test_version_1_0_payload_is_identical_to_previous_behaviour():
    job = prepare(Store(V1), CYCLE)
    assert set(values(job).values()) == {100.0}  # np.round(100.4), como antes del selector
    assert job['payload']['model']['version'] == 'calendar-test'


def test_version_2_0_applies_last_hour_ratio_and_is_recorded():
    store = Store(V2)
    job = prepare(store, CYCLE)
    got = values(job)
    assert got[('02300', '2026-09-14T15:15:00Z')] == round(100.4 * 150 / 100.4)
    assert got[('03000', '2026-09-14T16:00:00Z')] == 100.0
    assert store.recorded[0][0] == 'cyc_test' and store.recorded[0][1]['version'] == '2.0'
    assert job['payload']['model']['version'] == 'calendar-test'  # la reserva exige la versión registrada


@pytest.mark.parametrize('selected', [
    RuntimeError('tabla no existe'),
    {**V2, 'metodo': 'metodo_futuro'},
    {**V2, 'parametros': {'ventana_obs': 4, 'limite_min': float('nan'), 'limite_max': float('nan')}},
])
def test_any_selector_problem_sends_base_model(selected):
    store = Store(selected)
    job = prepare(store, CYCLE)
    assert set(values(job).values()) == {100.0}
    assert store.recorded[0][1]['version'] == '1.0' and 'fallback' in store.recorded[0][1]


def test_record_failure_never_blocks_and_lost_race_records_nothing():
    assert prepare(Store(V2, fail_record=True), CYCLE)['payload']
    winner = {'ciclo_id': 'cyc_test', 'client_run_id': 'other-runner-id', 'payload': {'predictions': []}, 'respuesta': None}
    store = Store(V2, existing=winner)
    assert prepare(store, CYCLE)['client_run_id'] == 'other-runner-id'
    assert store.recorded == []


class AttemptStore:
    def start_attempt(self, *a): pass
    def finish_attempt(self, attempt_id, result): self.finished = result
    def observe_cycle(self, cycle): pass
    def evaluate_deliveries(self): return {}
    def review_delivery_version(self): raise RuntimeError('rpc caída')


def test_review_failure_never_marks_attempt_as_error(monkeypatch):
    monkeypatch.setattr(automation, 'collect', lambda *a: {'inserted': 0})
    monkeypatch.setattr(automation, 'check_current_cycle', lambda *a: None)
    result = automation.execute(AttemptStore(), None)
    assert 'status' not in result
    assert result['version_review'] == {'status': 'error', 'error_type': 'RuntimeError'}
