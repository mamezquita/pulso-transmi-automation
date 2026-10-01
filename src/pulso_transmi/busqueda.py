"""Búsqueda exhaustiva de modelos con validación temporal.

Cortes (en ciclos horarios, del más antiguo al más reciente):
  … entrenamiento … | pliegue 1 | pliegue 2 | pliegue 3 | prueba final
Cada configuración se entrena con todo lo anterior a cada pliegue y se mide en él;
el puntaje de ajuste es el promedio de los tres. La mejor configuración de cada
familia se reentrena con todo lo anterior a la prueba final y se mide ahí, con la
fórmula del profesor: promedio por estación de 100·(1 − WAPE). La prueba final nunca
participa en el ajuste.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .model import evaluar
from .modelos import FEATURE_SETS, ModeloHistoria

FOLD_ORIGINS = 24  # un día de ciclos horarios por pliegue
CATEGORICAL = ['st']


def _cyclic(X):
    X = np.asarray(X, dtype=float)
    return np.column_stack([np.sin(2 * np.pi * X[:, 0] / 96), np.cos(2 * np.pi * X[:, 0] / 96),
                            np.sin(2 * np.pi * X[:, 1] / 7), np.cos(2 * np.pi * X[:, 1] / 7)])


def _scaled_pipeline(features, model, onehot=True):
    """Codificación para modelos sensibles a escala: estación one-hot, hora cíclica, normalización."""
    numeric = [f for f in features if f not in ('st', 'franja', 'dow')]
    parts = [('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric),
             ('ciclo', FunctionTransformer(_cyclic), ['franja', 'dow'])]
    if onehot:
        parts.append(('est', OneHotEncoder(handle_unknown='ignore'), ['st']))
    return make_pipeline(ColumnTransformer(parts), model)


def _imputed(model):
    return make_pipeline(SimpleImputer(strategy='median'), model)


# Familias: nombre -> (constructor(params, features) -> estimador, espacio de búsqueda)
FAMILIES = {
    'lightgbm': (lambda p, f: LGBMRegressor(verbose=-1, random_state=42, n_jobs=4, **p), {
        'objective': ['l1', 'l2', 'poisson', 'huber'], 'n_estimators': [200, 400, 800, 1500],
        'learning_rate': [0.01, 0.03, 0.05, 0.1], 'num_leaves': [15, 31, 63, 127],
        'min_child_samples': [20, 50, 100, 200], 'subsample': [0.6, 0.8, 1.0], 'subsample_freq': [1],
        'colsample_bytree': [0.6, 0.8, 1.0], 'reg_lambda': [0, 1, 5, 20]}),
    'bosque_aleatorio': (lambda p, f: _imputed(RandomForestRegressor(random_state=42, n_jobs=4, **p)), {
        'n_estimators': [200, 400], 'max_depth': [None, 12, 20], 'min_samples_leaf': [1, 5, 20],
        'max_features': [0.3, 0.6, 1.0]}),
    'arboles_extremos': (lambda p, f: _imputed(ExtraTreesRegressor(random_state=42, n_jobs=4, **p)), {
        'n_estimators': [200, 400], 'max_depth': [None, 12, 20], 'min_samples_leaf': [1, 5, 20],
        'max_features': [0.3, 0.6, 1.0]}),
    'gradient_boosting': (lambda p, f: HistGradientBoostingRegressor(random_state=42, **p), {
        'loss': ['absolute_error', 'squared_error', 'poisson'], 'learning_rate': [0.03, 0.06, 0.1],
        'max_iter': [300, 600, 1000], 'max_leaf_nodes': [15, 31, 63], 'l2_regularization': [0, 1, 5]}),
    'lineal_ridge': (lambda p, f: _scaled_pipeline(f, Ridge(**p)), {'alpha': [0.01, 0.1, 1, 10, 100]}),
    'knn': (lambda p, f: _scaled_pipeline(f, KNeighborsRegressor(n_jobs=4, **p)), {
        'n_neighbors': [5, 10, 20, 40, 80], 'weights': ['uniform', 'distance']}),
    'red_neuronal': (lambda p, f: _scaled_pipeline(f, MLPRegressor(random_state=42, early_stopping=True,
                                                                   max_iter=300, **p)), {
        'hidden_layer_sizes': [(32,), (64,), (64, 32), (128, 64)], 'alpha': [1e-4, 1e-3, 1e-2],
        'learning_rate_init': [1e-3, 3e-3]}),
}
COMMON = {'feature_set': list(FEATURE_SETS), 'objetivo': ['relativo', 'directo']}


def split(origins, folds=3):
    """Orígenes con objetivos completos -> (pliegues, prueba)."""
    n = FOLD_ORIGINS
    if len(origins) < n * (folds + 1) + 48:
        raise ValueError('Historia insuficiente para validación temporal')
    test = origins[-n:]
    fold_list = [origins[-n * (k + 2):-n * (k + 1)] for k in reversed(range(folds))]
    return fold_list, test


def score(real, pred, stations):
    m = evaluar(real, pred, stations)
    pos = real > 0
    return {'accuracy': m['Accuracy_macro_%'], 'wape': m['WAPE_macro'], 'mae': m['MAE'],
            'mape': float(np.mean(np.abs(real[pos] - pred[pos]) / real[pos]) * 100), 'n': int(len(real))}


class Corte:
    """Variables precalculadas de un corte temporal (independientes de los hiperparámetros)."""
    def __init__(self, actual, train_until, eval_origins, all_origins):
        train_o = [o for o in all_origins if o + pd.Timedelta(minutes=60) < train_until]
        self.shell = ModeloHistoria(None, 'perfil')
        self.X, self.y = self.shell.prepare(actual.loc[:train_until - pd.Timedelta(minutes=15)], train_o)
        self.Xe = self.shell._features(actual, eval_origins, set(self.shell.stations), (15, 30, 45, 60))
        self.re = np.array([actual.at[t, s] for t, s in zip(self.Xe.target_at, self.Xe.station_id)], dtype=float)


def _evaluate(corte, family, cfg):
    build, _ = FAMILIES[family]
    params = {k: v for k, v in cfg.items() if k not in COMMON}
    model = ModeloHistoria(build(params, FEATURE_SETS[cfg['feature_set']]), family, cfg['feature_set'],
                           cfg['objetivo'], cfg).adopt(corte.shell)
    model.fit_prepared(corte.X, corte.y)
    pred = model._predict_frame(corte.Xe)
    ok = ~np.isnan(corte.re)
    return model, score(corte.re[ok], np.round(pred[ok]), corte.Xe.station_id.to_numpy()[ok])


def cortes(actual, origins):
    folds, test = split(origins)
    return [Corte(actual, f[0], f, origins) for f in folds], Corte(actual, test[0], test, origins)


def sample(space, rng):
    return {k: v[rng.integers(len(v))] for k, v in {**COMMON, **space}.items()}


def search_family(cuts, family, trials=20, seed=42, time_budget=None, log=print):
    """Mejor configuración por promedio de los pliegues y su resultado en la prueba final."""
    folds, test = cuts
    rng = np.random.default_rng(seed)
    space = FAMILIES[family][1]
    seen, results, start = set(), [], time.time()
    for _ in range(trials * 3):
        if len(results) >= trials or (time_budget and time.time() - start > time_budget):
            break
        cfg = sample(space, rng)
        key = tuple(sorted((k, str(v)) for k, v in cfg.items()))
        if key in seen:
            continue
        seen.add(key)
        fold_scores = [_evaluate(c, family, cfg)[1]['accuracy'] for c in folds]
        results.append((float(np.mean(fold_scores)), cfg, fold_scores))
        log(f'  {family} prueba {len(results)}: pliegues {np.round(fold_scores, 1)} -> {np.mean(fold_scores):.2f}')
    best_cv, best_cfg, best_folds = max(results, key=lambda r: r[0])
    model, test_score = _evaluate(test, family, best_cfg)
    return {'familia': family, 'config': best_cfg, 'cv_accuracy': best_cv, 'pliegues': best_folds,
            'prueba': test_score, 'pruebas_realizadas': len(results), 'segundos': round(time.time() - start, 1)}, model


def current_pipeline_score(actual, model_pivot, test, ratio_window=4, lo=0.5, hi=2.0):
    """Lo que se entrega hoy (modelo activo + corrección de la versión activa) en la prueba."""
    from . import backtest as bt
    frame = bt.simulate(actual, model_pivot, test, window=ratio_window) if ratio_window else bt.simulate(actual, model_pivot, test)
    pred = np.round(frame.corrected if ratio_window else frame.model)
    return score(frame.real.to_numpy(float), pred.to_numpy(float), frame.station_id.to_numpy())
