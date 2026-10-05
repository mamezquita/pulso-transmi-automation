from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pulso_transmi import busqueda as bq, reentreno, shadow

CUTOFF = '2026-09-14T15:00:00Z'
CYCLE = {'cycle_id': 'cyc_x', 'data_cutoff': CUTOFF,
         'targets': [{'station_id': '02300', 'target_at': '2026-09-14T15:15:00Z'},
                     {'station_id': '03000', 'target_at': '2026-09-14T16:00:00Z'}]}


class ShadowStore:
    def __init__(self, with_shadow=True):
        self.saved, self.with_shadow = [], with_shadow
    def shadow_versions(self):
        return [{'version': '3.0', 'metodo': 'modelo_sombra', 'parametros': {'nombre': 'combinacion'}, 'estado': 'sombra'}]
    def shadow_done(self, c): return set()
    def active_model(self): return {'sha256': 'activo', 'version': 'm1-lightgbm-r26'}
    def shadow_model(self, name):
        return {'sha256': 'sombra', 'version': 'sombra-combinacion-x', 'familia': 'combinacion'} if self.with_shadow else None
    def load_model(self, rec):
        value = 100.0 if rec['sha256'] == 'activo' else 120.0
        return SimpleNamespace(lookback_obs=0, predict=lambda d, v=value: np.full(len(d), v))
    def save_shadow(self, row): self.saved.append(row)


def test_shadow_version_predicts_with_the_shadow_model_and_records_its_sha():
    store = ShadowStore()
    out = shadow.run_shadow(store, CYCLE)
    row = store.saved[0]
    assert out['versions'] == ['3.0'] and row['modelo_sha256'] == 'sombra'
    assert [p['value'] for p in row['predicciones']] == [120.0, 120.0]


def test_shadow_version_waits_quietly_until_a_shadow_model_exists():
    store = ShadowStore(with_shadow=False)
    assert shadow.run_shadow(store, CYCLE)['versions'] == [] and store.saved == []


def test_publishing_a_new_shadow_model_replaces_and_deletes_the_previous_artifact(monkeypatch):
    deleted, stored = [], []
    store = SimpleNamespace(
        shadow_model=lambda name: {'sha256': 'viejo', 'storage_path': 'viejo/model.joblib'},
        set_shadow_model=stored.append, client=SimpleNamespace(delete=deleted.append))
    monkeypatch.setattr(reentreno, 'upload_artifact', lambda *a, **k: ('nuevo', {}))
    model = SimpleNamespace(familia='combinacion')
    reentreno.publish_shadow(store, model, {'config': {'miembros': []}, 'cv_accuracy': 70, 'prueba': {'accuracy': 69}})
    assert stored[0]['modelo_sha256'] == 'nuevo' and stored[0]['metricas']['accuracy'] == 69
    assert deleted == ['/storage/v1/object/pulso-models/viejo/model.joblib']


def result(fam, acc):
    return {'familia': fam, 'config': {'f': fam}, 'cv_accuracy': acc, 'pliegues': [acc] * 3,
            'prueba': {'accuracy': acc, 'wape': 0.2, 'mae': 1.0, 'mape': 1.0, 'n': 10},
            'pruebas_realizadas': 1, 'segundos': 1}


@pytest.fixture
def wired(monkeypatch):
    calls = {'search': [], 'saved': [], 'published': []}
    monkeypatch.setattr(bq, 'search_family', lambda cuts, fam, **k: calls['search'].append(fam) or (result(fam, 70), None))
    monkeypatch.setattr(bq, 'search_combination', lambda cuts, results, **k: result('combinacion', 72))
    monkeypatch.setattr(bq, 'build_model', lambda fam, cfg: SimpleNamespace(fit=lambda a, o: SimpleNamespace(familia=fam)))
    monkeypatch.setattr(reentreno, 'save_candidate', lambda store, rid, r, chosen=False: calls['saved'].append((r['familia'], r['config'].get('sombra'), chosen)))
    monkeypatch.setattr(reentreno, 'publish_shadow', lambda store, model, r: calls['published'].append(r['familia']) or 'sombra-v')
    return calls


def test_when_lightgbm_passes_the_threshold_other_families_are_searched_only_in_shadow(wired):
    out = reentreno.shadow_search(None, 1, None, None, None, result('lightgbm', 86), [], 'lightgbm', trials=8, budget=60)
    assert out == 'sombra-v' and wired['search'] == list(reentreno.SHADOW_FAMILIES)
    assert all(s is True and c is False for _, s, c in wired['saved'])  # nunca elegidos
    assert wired['published'] == ['combinacion']


def test_existing_combination_is_reused_and_a_chosen_combination_is_not_shadowed(wired):
    others = [result('gradient_boosting', 70), result('combinacion', 72)]
    reentreno.shadow_search(None, 1, None, None, None, result('lightgbm', 70), others, 'lightgbm', trials=8, budget=60)
    assert wired['search'] == [] and wired['saved'] == [] and wired['published'] == ['combinacion']
    wired['published'].clear()
    assert reentreno.shadow_search(None, 1, None, None, None, result('lightgbm', 70), others, 'combinacion',
                                   trials=8, budget=60) == 'sin_sombra'
    assert wired['published'] == []


def test_shadow_refresh_skips_without_model_or_new_data(monkeypatch):
    assert reentreno.refresh_shadow(SimpleNamespace(shadow_model=lambda n: None)) == 'sin_modelo_sombra'
    idx = pd.date_range('2026-09-14', periods=8, freq='15min', tz='UTC')
    monkeypatch.setattr(reentreno, 'load_data', lambda: (pd.DataFrame({'A': range(8)}, index=idx), []))
    store = SimpleNamespace(shadow_model=lambda n: {'familia': 'combinacion', 'config': {}},
                            load_model=lambda rec: SimpleNamespace(end=idx[-1].isoformat()))
    assert reentreno.refresh_shadow(store) == 'al_dia'
