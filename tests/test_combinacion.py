import io
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

from pulso_transmi import busqueda
from pulso_transmi.modelos import ModeloCombinado, ModeloHistoria
from test_modelos import IDX, history_rows, matrix


def test_combined_model_averages_members_and_survives_serialization():
    actual = matrix()
    origins = [t for t in IDX[100:-8] if t.minute == 0]
    a = ModeloHistoria(LinearRegression(), 'lineal', 'reciente_ayer', 'relativo')
    b = ModeloHistoria(Ridge(alpha=10), 'ridge', 'reciente', 'directo')
    combo = ModeloCombinado([a, b]).fit(actual, origins)
    cutoff = IDX[-20]
    targets = pd.DataFrame({'station_id': ['A', 'B'],
                            'target_at': [(cutoff + pd.Timedelta(minutes=m)).isoformat() for m in (15, 60)]})
    hist = history_rows(actual, cutoff)
    expected = (a.predict_at(targets, hist, cutoff) + b.predict_at(targets, hist, cutoff)) / 2
    assert np.allclose(combo.predict_at(targets, hist, cutoff), expected)
    assert combo.end == a.end and combo.lookback_obs == a.lookback_obs and combo.n_train == a.n_train
    buf = io.BytesIO(); joblib.dump(combo, buf); buf.seek(0)
    assert np.allclose(joblib.load(buf).predict_at(targets, hist, cutoff), expected)


def test_build_model_rebuilds_combination_from_member_configs():
    cfg = {'miembros': [{'familia': 'lineal_ridge', 'config': {'feature_set': 'reciente', 'objetivo': 'directo', 'alpha': 1}},
                        {'familia': 'knn', 'config': {'feature_set': 'calendario', 'objetivo': 'relativo',
                                                      'n_neighbors': 5, 'weights': 'uniform'}}]}
    model = busqueda.build_model(busqueda.COMBINATION, cfg)
    assert isinstance(model, ModeloCombinado)
    assert [m.familia for m in model.miembros] == ['lineal_ridge', 'knn']


def test_combination_picks_best_subset_by_folds_and_scores_it_on_test(monkeypatch):
    # Real = 100. 'alto' predice 120, 'bajo' 80, 'malo' 150: alto+bajo se compensan.
    preds = {'alto': 120.0, 'bajo': 80.0, 'malo': 150.0}
    cut = lambda: SimpleNamespace(re=np.full(4, 100.0), Xe=pd.DataFrame({'station_id': ['A', 'A', 'B', 'B']}))
    folds, test = [cut(), cut(), cut()], cut()
    monkeypatch.setattr(busqueda, '_evaluate', lambda c, f, cfg: (
        SimpleNamespace(_predict_frame=lambda X: np.full(len(X), preds[f])), None))
    results = [{'familia': f, 'config': {'f': f}, 'cv_accuracy': acc} for f, acc in (('alto', 80), ('bajo', 80), ('malo', 50))]
    out = busqueda.search_combination((folds, test), results, log=lambda m: None)
    assert [m['familia'] for m in out['config']['miembros']] == ['alto', 'bajo']
    assert out['familia'] == 'combinacion' and out['prueba']['accuracy'] == 100.0
    assert out['pruebas_realizadas'] == 4  # 3 pares + el trío


def test_combination_needs_at_least_two_families():
    assert busqueda.search_combination(([], None), [{'familia': 'lightgbm', 'cv_accuracy': 70, 'config': {}}]) is None


def test_active_model_that_saw_the_test_day_is_retrained_without_it(monkeypatch):
    from pulso_transmi import reentreno
    test_origins = [pd.Timestamp('2026-09-19T00:00:00Z')]
    calls = []
    monkeypatch.setattr(busqueda, 'evaluate_config', lambda cut, fam, cfg: calls.append((fam, cfg)) or {'accuracy': 60.0})
    monkeypatch.setattr(reentreno, 'current_score', lambda *a: {'accuracy': 85.0})
    leaked = SimpleNamespace(familia='lightgbm', config={'feature_set': 'reciente'}, end='2026-09-19T10:30:00-05:00')
    assert reentreno.fair_current_score(None, leaked, None, 'corte', test_origins)['accuracy'] == 60.0
    assert calls == [('lightgbm', {'feature_set': 'reciente'})]
    calendar = SimpleNamespace(end='2026-09-11T11:00:00-05:00')  # modelo de calendario, sin fuga
    assert reentreno.fair_current_score(None, calendar, None, 'corte', test_origins)['accuracy'] == 85.0


def test_recency_weights_halve_every_half_life():
    from pulso_transmi.modelos import recency_weights
    origins = pd.to_datetime(['2026-09-17T00:00Z', '2026-09-18T00:00Z', '2026-09-19T00:00Z'])
    assert np.allclose(recency_weights(origins, 1), [0.25, 0.5, 1.0])
    assert recency_weights(origins, None) is None


def test_weights_reach_the_final_step_and_are_skipped_when_unsupported():
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from pulso_transmi.modelos import fit_weighted
    X = np.array([[0.0], [1.0], [2.0], [3.0]]); y = np.array([0.0, 0.0, 10.0, 10.0])
    w = np.array([1.0, 1.0, 1e-6, 1e-6])
    weighted = fit_weighted(make_pipeline(StandardScaler(), Ridge(alpha=1e-6)), X, y, w)
    plain = make_pipeline(StandardScaler(), Ridge(alpha=1e-6)).fit(X, y)
    assert abs(weighted.predict([[1.0]])[0]) < abs(plain.predict([[1.0]])[0])  # los pesos sí cuentan
    fit_weighted(KNeighborsRegressor(n_neighbors=2), X, y, w)  # sin sample_weight: entrena sin pesos
