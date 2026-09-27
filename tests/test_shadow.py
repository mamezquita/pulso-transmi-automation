import numpy as np
import pandas as pd
import pytest

from pulso_transmi import automation
from pulso_transmi.shadow import ratio_ultima_hora, run_shadow

CUTOFF = '2026-09-14T15:00:00Z'
CYCLE = {'cycle_id': 'cyc_t', 'data_cutoff': CUTOFF, 'targets': [
    {'station_id': s, 'target_at': t, 'horizon_minutes': h}
    for s in ('A', 'B') for t, h in (('2026-09-14T15:15:00Z', 15), ('2026-09-14T16:00:00Z', 60))]}
V1 = {'version': '1.0', 'metodo': 'modelo_base', 'parametros': {}, 'estado': 'sombra'}
V2 = {'version': '2.0', 'metodo': 'ratio_ultima_hora', 'estado': 'sombra',
      'parametros': {'ventana_obs': 4, 'limite_min': 0.5, 'limite_max': 2.0}}


class Model:
    def predict(self, df):
        return np.full(len(df), 100.0)


def history(a=150, b=100, extra_future=True):
    times = pd.date_range('2026-09-14T14:15:00Z', CUTOFF, freq='15min')
    rows = [{'station_id': s, 'observed_at': t.isoformat(), 'demand': v}
            for t in times for s, v in (('A', a), ('B', b))]
    if extra_future:  # datos posteriores al corte que la sombra jamás debe usar
        rows += [{'station_id': 'A', 'observed_at': '2026-09-14T15:15:00Z', 'demand': 10_000}]
    return rows


class Store:
    def __init__(self, versions=(V1, V2), done=(), rows=None):
        self.versions = list(versions); self.done = set(done); self.saved = []; self.loads = 0
        self.rows = history() if rows is None else rows
    def shadow_versions(self): return self.versions
    def shadow_done(self, cycle_id): return self.done
    def active_model(self): return {'sha256': 'f' * 64}
    def load_model(self, record): self.loads += 1; return Model()
    def observations_window(self, start, end): return self.rows
    def save_shadow(self, row): self.saved.append(row)


def by_version(store):
    return {r['version']: {(p['station_id'], p['target_at']): p['value'] for p in r['predicciones']}
            for r in store.saved}


def test_control_equals_model_and_ratio_uses_last_hour_only():
    store = Store()
    out = run_shadow(store, CYCLE)
    assert out == {'status': 'computed', 'versions': ['1.0', '2.0'], 'unknown_method': []}
    preds = by_version(store)
    assert set(preds['1.0'].values()) == {100.0}
    # A: 150/100 = 1.5 con la última hora; el valor futuro de 10.000 no cuenta.
    assert preds['2.0'][('A', '2026-09-14T15:15:00Z')] == 150.0
    assert preds['2.0'][('B', '2026-09-14T16:00:00Z')] == 100.0
    assert store.saved[1]['diagnostico']['ratios'] == {'A': 1.5, 'B': 1.0}


def test_ratio_is_clipped_and_missing_station_falls_back_to_one():
    targets = pd.DataFrame(CYCLE['targets'])[['station_id', 'target_at']]
    hist = pd.DataFrame(history(a=1000, b=100, extra_future=False))
    hist = hist[hist.station_id == 'A']
    hist['observed_at'] = pd.to_datetime(hist.observed_at, utc=True)
    values, diag = ratio_ultima_hora(np.full(4, 100.0), targets, hist, Model(),
                                     pd.Timestamp(CUTOFF), **V2['parametros'])
    assert diag['ratios'] == {'A': 2.0, 'B': 1.0} and diag['sin_datos'] == ['B']
    assert list(values) == [200.0, 200.0, 100.0, 100.0]


def test_incomplete_last_hour_is_not_trusted():
    rows = [r for r in history(extra_future=False) if not (r['station_id'] == 'A' and r['observed_at'].startswith('2026-09-14T14:45'))]
    store = Store(versions=[V2], rows=rows)
    run_shadow(store, CYCLE)
    assert store.saved[0]['diagnostico']['sin_datos'] == ['A']


def test_already_computed_versions_do_not_reload_model():
    store = Store(done={'1.0', '2.0'})
    assert run_shadow(store, CYCLE)['status'] == 'up_to_date'
    assert store.loads == 0 and store.saved == []


def test_unknown_method_is_skipped_not_fatal():
    future = {'version': '3.0', 'metodo': 'metodo_futuro', 'parametros': {}, 'estado': 'sombra'}
    store = Store(versions=[V1, future])
    out = run_shadow(store, CYCLE)
    assert out['versions'] == ['1.0'] and out['unknown_method'] == ['3.0']


def test_paused_or_retired_versions_are_ignored():
    store = Store(versions=[V1, {**V2, 'estado': 'retirada'}])
    assert run_shadow(store, CYCLE)['versions'] == ['1.0']


class AttemptStore:
    def start_attempt(self, *a): pass
    def finish_attempt(self, attempt_id, result): self.finished = result
    def observe_cycle(self, cycle): pass
    def evaluate_deliveries(self): return {}


def test_shadow_failure_never_affects_submission(monkeypatch):
    monkeypatch.setattr(automation, 'collect', lambda *a: {'inserted': 0})
    monkeypatch.setattr(automation, 'check_current_cycle', lambda *a: CYCLE)
    monkeypatch.setattr(automation, 'forecast', lambda *a, **k: {'status': 'accepted', 'submitted': True})
    def boom(*a): raise RuntimeError('supabase caído')
    monkeypatch.setattr(automation, 'run_shadow', boom)
    result = automation.execute(AttemptStore(), None)
    assert result['forecast']['submitted'] and 'status' not in result
    assert result['shadow'] == {'status': 'error', 'error_type': 'RuntimeError'}


def test_shadow_runs_after_submission(monkeypatch):
    order = []
    monkeypatch.setattr(automation, 'collect', lambda *a: {'inserted': 0})
    monkeypatch.setattr(automation, 'check_current_cycle', lambda *a: CYCLE)
    monkeypatch.setattr(automation, 'forecast', lambda *a, **k: order.append('forecast') or {'submitted': True})
    monkeypatch.setattr(automation, 'run_shadow', lambda *a: order.append('shadow') or {'status': 'computed'})
    automation.execute(AttemptStore(), None)
    assert order == ['forecast', 'shadow']
