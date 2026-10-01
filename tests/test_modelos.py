import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from pulso_transmi import automation, busqueda
from pulso_transmi.modelos import ModeloHistoria
from pulso_transmi.shadow import base_predictions

IDX = pd.date_range('2026-09-01', periods=96 * 6, freq='15min', tz='UTC')


def matrix(seed=0):
    rng = np.random.default_rng(seed)
    base = 200 + 80 * np.sin(np.arange(len(IDX)) * 2 * np.pi / 96)
    return pd.DataFrame({'A': base + rng.normal(0, 5, len(IDX)), 'B': 2 * base + rng.normal(0, 5, len(IDX))}, index=IDX)


def fitted():
    actual = matrix()
    origins = [t for t in IDX[100:-8] if t.minute == 0]
    return actual, ModeloHistoria(LinearRegression(), 'lineal', 'reciente_ayer', 'relativo').fit(actual, origins)


def history_rows(actual, until):
    part = actual.loc[:until]
    return pd.DataFrame([{'station_id': s, 'observed_at': t, 'demand': v}
                         for t, row in part.iterrows() for s, v in row.items()])


def test_predict_at_uses_only_history_up_to_cutoff_and_keeps_target_order():
    actual, model = fitted()
    cutoff = IDX[-20]
    targets = pd.DataFrame({'station_id': ['B', 'A', 'A'],
                            'target_at': [(cutoff + pd.Timedelta(minutes=m)).isoformat() for m in (60, 15, 30)]})
    hist = history_rows(actual, cutoff)
    first = model.predict_at(targets, hist, cutoff)
    future = pd.concat([hist, pd.DataFrame([{'station_id': 'A', 'observed_at': cutoff + pd.Timedelta(minutes=15), 'demand': 1e6}])])
    assert np.allclose(first, model.predict_at(targets, future, cutoff))
    assert first[0] > first[1]  # B (el doble de A) primero, como pidió el contrato
    assert (first > 0).all()


def test_unknown_horizon_is_rejected_and_calendar_predict_is_profile():
    actual, model = fitted()
    cutoff = IDX[-20]
    bad = pd.DataFrame({'station_id': ['A'], 'target_at': [(cutoff + pd.Timedelta(minutes=90)).isoformat()]})
    with pytest.raises(ValueError):
        model.predict_at(bad, history_rows(actual, cutoff), cutoff)
    prof = model.predict(pd.DataFrame({'station_id': ['A'], 'observed_at': [IDX[10].isoformat()]}))
    assert 100 < prof[0] < 300


class Store:
    def __init__(self, actual): self.actual, self.asked = actual, []
    def observations_window(self, start, end):
        self.asked.append((start, end))
        part = self.actual.loc[(self.actual.index > start) & (self.actual.index <= end)]
        return history_rows(part, end).to_dict('records')


def test_forecaster_sends_history_only_to_history_models():
    actual, model = fitted()
    cutoff = IDX[-20]
    targets = pd.DataFrame({'station_id': ['A'], 'target_at': [(cutoff + pd.Timedelta(minutes=15)).isoformat()]})
    store = Store(actual)
    assert base_predictions(store, model, targets, cutoff).shape == (1,)
    assert store.asked and store.asked[0][1] == cutoff
    class Calendar:
        def predict(self, df): return np.full(len(df), 7.0)
    store2 = Store(actual)
    assert list(base_predictions(store2, Calendar(), targets, cutoff)) == [7.0] and store2.asked == []


class AttemptStore:
    def start_attempt(self, *a): pass
    def finish_attempt(self, attempt_id, result): self.finished = result
    def observe_cycle(self, cycle): pass
    def evaluate_deliveries(self): return {'drift': {'status': 'evaluado', 'decision': 'reentrenar'}}
    def review_delivery_version(self): return {}


def test_each_attempt_records_its_stages(monkeypatch):
    monkeypatch.setattr(automation, 'collect', lambda *a: {'inserted': 3, 'quarantined': 0})
    monkeypatch.setattr(automation, 'check_current_cycle', lambda *a: None)
    result = automation.execute(AttemptStore(), None)
    etapas = {e['etapa']: e for e in result['etapas']}
    assert list(etapas) == ['colector', 'envio', 'drift']
    assert etapas['colector']['resultado'] == {'insertados': 3, 'cuarentena': 0}
    assert etapas['drift']['resultado']['decision'] == 'reentrenar'
    assert all(e['estado'] == 'ok' and e['ms'] >= 0 for e in result['etapas'])


def test_failed_stage_is_marked_as_error(monkeypatch):
    def boom(*a): raise RuntimeError('caído')
    monkeypatch.setattr(automation, 'collect', boom)
    monkeypatch.setattr(automation, 'check_current_cycle', lambda *a: None)
    result = automation.execute(AttemptStore(), None)
    assert result['etapas'][0] == {**result['etapas'][0], 'etapa': 'colector', 'estado': 'error',
                                   'resultado': {'error_type': 'RuntimeError'}}


def test_temporal_split_keeps_order_and_holds_out_the_latest_day():
    origins = list(pd.date_range('2026-09-01', periods=24 * 8, freq='h', tz='UTC'))
    folds, test = busqueda.split(origins)
    assert test == origins[-24:]
    assert folds[0][-1] < folds[1][0] and folds[2][-1] < test[0]
    with pytest.raises(ValueError):
        busqueda.split(origins[:100])
