from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from pulso_transmi import reentreno
from pulso_transmi.modelos import ModeloHistoria
from test_modelos import IDX, matrix

CFG = {'feature_set': 'reciente', 'objetivo': 'relativo', 'alpha': 1.0}


def active_model(until):
    actual = matrix()
    origins = [t for t in IDX[100:-8] if t.minute == 0 and t + pd.Timedelta(minutes=60) <= until]
    return ModeloHistoria(Ridge(alpha=1.0), 'lineal_ridge', 'reciente', 'relativo', CFG).fit(actual.loc[:until], origins)


class Store:
    def __init__(self, model):
        self.model = model; self.posts = []; self.patches = []; self.deleted = []
        self.client = SimpleNamespace(delete=lambda path: self.deleted.append(path) or SimpleNamespace(status_code=200))
    def active_model(self): return {'sha256': 'activo', 'version': 'm1-lineal_ridge-r2'}
    def load_model(self, record): return self.model
    def rows(self, table, **k):
        if table == 'modelo_activo_detalle':
            return [{'version_mayor': 1, 'revision': 2, 'metricas': {'cv_accuracy': 70.0, 'accuracy': 65.0}}]
        if table == 'version_modelo' and 'revision' in k.get('select', ''):
            return [{'revision': 4}]
        if table == 'version_modelo':
            return [{'modelo_sha256': 'viejo'}, {'modelo_sha256': 'activo'}]
        if table == 'registro_modelo_operativo':
            return [{'storage_path': k['sha256'][3:] + '/model.joblib'}]
        return []
    def request(self, method, path, **k):
        (self.posts if method == 'POST' else self.patches).append((path, k.get('json')))
        return SimpleNamespace(json=lambda: [{'reentreno_id': 99}])


@pytest.fixture
def wired(monkeypatch):
    actual = matrix()
    origins = [t for t in IDX[100:-8] if t.minute == 0]
    published, activated = [], []
    monkeypatch.setattr(reentreno, 'load_data', lambda: (actual, origins))
    monkeypatch.setattr(reentreno, 'publish', lambda store, model, major, rev, fam, result, motivo:
                        published.append((model, major, rev, fam, result)) or ('abc' * 20, f'm{major}-{fam}-r{rev}'))
    monkeypatch.setattr(reentreno, 'wait_safe_window', lambda store, max_wait: True)
    monkeypatch.setattr(reentreno, 'activate', lambda store, major, rev, motivo: activated.append((major, rev)) or {})
    return published, activated


def test_refresh_retrains_same_config_with_newer_data_as_next_revision(wired):
    published, activated = wired
    old = active_model(IDX[300])
    store = Store(old)
    out = reentreno.refresh(store)
    model, major, rev, fam, result = published[0]
    assert out['status'] == 'refrescado' and (major, rev, fam) == (1, 5, 'lineal_ridge')
    assert pd.Timestamp(model.end) > pd.Timestamp(old.end) and model.config == CFG
    assert result['config']['origen'] == 'refresco' and result['config']['refresco_de'] == '1 r2'
    assert activated == [(1, 5)]
    assert store.posts[0][1]['fase'] == 'refresco'
    assert store.patches[-1][1]['estado'] == 'completado'
    assert store.deleted == ['/storage/v1/object/pulso-models/viejo/model.joblib']  # nunca el activo


def test_refresh_skipped_when_model_is_up_to_date_or_has_no_config(wired):
    published, _ = wired
    assert reentreno.refresh(Store(active_model(IDX[-3])))['status'] == 'sin_refresco'
    assert reentreno.refresh(Store(SimpleNamespace(end='2026-09-01T00:00:00Z')))['status'] == 'sin_refresco'
    assert published == []


def test_refresh_failure_is_recorded_and_raised(wired, monkeypatch):
    monkeypatch.setattr(reentreno, 'publish', lambda *a: (_ for _ in ()).throw(RuntimeError('storage caído')))
    store = Store(active_model(IDX[300]))
    with pytest.raises(RuntimeError):
        reentreno.refresh(store)
    assert store.patches[-1][1]['estado'] == 'error'


def test_session_runs_once_per_hour_at_minute_10_until_time_runs_out(monkeypatch):
    now = [datetime(2026, 10, 2, 22, 40, tzinfo=timezone.utc)]
    calls = []
    monkeypatch.setattr(reentreno, 'run', lambda **kw: calls.append(now[0]) or {'status': 'refrescado'})
    sleep = lambda secs: now.__setitem__(0, now[0] + pd.Timedelta(seconds=secs))
    out = reentreno.session(300, sleep=sleep, clock=lambda: now[0])
    assert calls[0].minute == 40 and all(c.minute == 10 for c in calls[1:])
    assert out['iteraciones'] == len(calls) == 5  # 22:40, 23:10, 00:10, 01:10, 02:10; la de 03:10 no cabe


def test_session_survives_an_iteration_error(monkeypatch):
    now = [datetime(2026, 10, 2, 22, 40, tzinfo=timezone.utc)]
    monkeypatch.setattr(reentreno, 'run', lambda **kw: (_ for _ in ()).throw(ValueError('x')))
    sleep = lambda secs: now.__setitem__(0, now[0] + pd.Timedelta(seconds=secs))
    assert reentreno.session(200, sleep=sleep, clock=lambda: now[0])['iteraciones'] == 3
