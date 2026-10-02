from types import SimpleNamespace

import numpy as np
import pytest

from pulso_transmi.forecaster import FALLBACK_CORRECTION, FALLBACK_MODEL, prepare
from test_delivery_version import CYCLE, V1, Store, values

V23 = {'version': '2.3', 'metodo': 'ratio_ultima_hora', 'estado': 'sombra',
       'parametros': {'ventana_obs': 4, 'limite_min': 0.5, 'limite_max': 2.0}}
CALENDAR = {'sha256': 'c' * 64, 'version': 'calendar-lgbm-v1-portable',
            'metadata': {'training_data_end': '2026-09-08T00:00:00Z', 'trained_at': '2026-09-20T00:00:00Z'}}


def broken_history_model(end):
    def predict_at(targets, history, cutoff): raise ValueError('horizonte desconocido')
    return SimpleNamespace(end=end, lookback_obs=8, predict_at=predict_at)


class FallbackStore(Store):
    """Modelo activo con historia que falla de la forma indicada; respaldo = calendario."""
    def __init__(self, failure='predict', fallback_failure=False, active_is_calendar=False):
        super().__init__(V1)
        self.failure = failure; self.fallback_failure = fallback_failure; self.reserved = []
        self.versions = {'2.3': V23}
        if active_is_calendar: self.meta = CALENDAR

    def fallback_model(self, major, revision):
        assert (major, revision) == FALLBACK_MODEL
        return CALENDAR

    def shadow_version(self, version): return self.versions[version]

    def load_model(self, r):
        end = r['metadata']['training_data_end']
        if r is CALENDAR:
            if self.fallback_failure: raise RuntimeError('storage caído')
            return SimpleNamespace(end=end, predict=lambda d: np.full(len(d), 100.4))
        if self.failure == 'load': raise ValueError('Versión incompatible: instalar lightgbm==4.7.0')
        if self.failure == 'nan':
            return SimpleNamespace(end=end, lookback_obs=8, predict_at=lambda t, h, c: np.full(len(t), np.nan))
        if self.failure == 'predict': return broken_history_model(end)
        return SimpleNamespace(end=end, lookback_obs=8, predict_at=lambda t, h, c: np.full(len(t), 120.0))

    def reserve(self, payload, sha, fallback=False):
        self.reserved.append((sha, fallback))
        return super().reserve(payload, sha)


@pytest.mark.parametrize('failure', ['predict', 'load', 'nan'])
def test_active_model_failure_sends_calendar_model_with_correction(failure):
    store = FallbackStore(failure)
    job = prepare(store, CYCLE)
    assert job['payload']['model']['version'] == 'calendar-lgbm-v1-portable'
    assert job['payload']['model']['training_data_end'] == CALENDAR['metadata']['training_data_end']
    assert len(job['payload']['predictions']) == 4 and all(v > 0 for v in values(job).values())
    assert store.reserved == [(CALENDAR['sha256'], True)]
    selection = store.recorded[0][1]
    assert selection['version'] == FALLBACK_CORRECTION
    assert selection['respaldo']['modelo_fallido'] == 'calendar-test'


def test_correction_is_applied_on_top_of_calendar_model():
    job = prepare(FallbackStore('predict'), CYCLE)
    # ratio de la última hora: 150 real vs 100.4 del modelo en la estación 02300
    assert values(job)[('02300', '2026-09-14T15:15:00Z')] == round(100.4 * 150 / 100.4)


def test_healthy_active_model_never_uses_fallback():
    store = FallbackStore('none')
    job = prepare(store, CYCLE)
    assert job['payload']['model']['version'] == 'calendar-test'
    assert set(values(job).values()) == {120.0}
    assert store.reserved == [('a' * 64, False)]
    assert 'respaldo' not in store.recorded[0][1]


def test_fallback_failure_raises_without_reserving():
    store = FallbackStore('predict', fallback_failure=True)
    with pytest.raises(ValueError, match='horizonte desconocido') as err:
        prepare(store, CYCLE)
    assert isinstance(err.value.__cause__, RuntimeError)  # la falla del respaldo queda encadenada
    assert store.reserved == []


def test_calendar_model_failing_is_not_retried_with_itself():
    store = FallbackStore(active_is_calendar=True, fallback_failure=True)
    with pytest.raises(RuntimeError, match='storage caído'):
        prepare(store, CYCLE)
    assert store.reserved == []
