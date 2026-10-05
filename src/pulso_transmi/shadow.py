"""Modo sombra versionado: calcula pronósticos alternativos y los guarda aparte.

Nunca reserva ni envía. Cada versión (tabla version_sombra) nombra un método de
METHODS y sus parámetros; ambos son inmutables en la base. Solo se usan datos con
observado_en <= data_cutoff, igual que el envío real.
"""
import os

import numpy as np
import pandas as pd

from .prepare_submission import timestamp

STEP = pd.Timedelta(minutes=15)


def modelo_base(base, targets, history, model, cutoff):
    """Control: el modelo activo tal como se envía (debe coincidir con lo entregado)."""
    return np.round(base), {}


def ratio_ultima_hora(base, targets, history, model, cutoff, *, ventana_obs=4,
                      limite_min=0.5, limite_max=2.0):
    """Modelo × (Σ real / Σ modelo) de las últimas `ventana_obs` observaciones por estación."""
    since = cutoff - ventana_obs * STEP
    past = history[(history.observed_at > since) & (history.observed_at <= cutoff)]
    ratios, fallback = {}, []
    if not past.empty:
        past = past.assign(model=model.predict(past[['station_id', 'observed_at']]))
        for sid, g in past.groupby('station_id'):
            if len(g) == ventana_obs and g.model.sum() > 0:
                ratios[sid] = float(np.clip(g.demand.sum() / g.model.sum(), limite_min, limite_max))
    for sid in targets.station_id.unique():
        if sid not in ratios:
            ratios[sid] = 1.0
            fallback.append(sid)
    factor = targets.station_id.map(ratios).to_numpy(dtype=float)
    return np.round(np.maximum(base * factor, 0)), {
        'ratios': {k: round(v, 4) for k, v in sorted(ratios.items())}, 'sin_datos': sorted(fallback)}


def _ratios(history, model, cutoff, k, lo, hi):
    """Razón real/modelo de las últimas k observaciones por estación (solo ventanas completas)."""
    past = history[(history.observed_at > cutoff - k * STEP) & (history.observed_at <= cutoff)]
    out = {}
    if not past.empty:
        past = past.assign(model=model.predict(past[['station_id', 'observed_at']]))
        for sid, g in past.groupby('station_id'):
            if len(g) == k and g.model.sum() > 0:
                out[sid] = float(np.clip(g.demand.sum() / g.model.sum(), lo, hi))
    return out


def ratio_conjunto(base, targets, history, model, cutoff, *, ventanas=(2, 4), limite_min=0.33,
                   limite_max=3.0, ventana_obs=None):
    """Modelo × promedio de las razones con varias ventanas (menos ruido que una sola)."""
    per_window = [_ratios(history, model, cutoff, int(k), limite_min, limite_max) for k in ventanas]
    ratios, fallback = {}, []
    for sid in targets.station_id.unique():
        vals = [r[sid] for r in per_window if sid in r]
        ratios[sid] = float(np.mean(vals)) if vals else 1.0
        if not vals:
            fallback.append(sid)
    factor = targets.station_id.map(ratios).to_numpy(dtype=float)
    return np.round(np.maximum(base * factor, 0)), {
        'ratios': {k: round(v, 4) for k, v in sorted(ratios.items())}, 'sin_datos': sorted(fallback)}


def ratio_estacional(base, targets, history, model, cutoff, *, ventana_obs=2, peso=0.2,
                     limite_min=0.33, limite_max=3.0, historial_obs=None):
    """(1-peso) × modelo corregido + peso × (ayer a la misma hora × hoy/ayer en el corte)."""
    ratios = _ratios(history, model, cutoff, int(ventana_obs), limite_min, limite_max)
    level = history.set_index(['station_id', 'observed_at']).demand
    yday_cut = cutoff - pd.Timedelta(days=1)
    corrected, seasonal_used, fallback = [], 0, []
    for sid, t, b in zip(targets.station_id, pd.to_datetime(targets.target_at, utc=True), base):
        r = ratios.get(sid)
        if r is None and sid not in fallback:
            fallback.append(sid)
        value = b * (r if r is not None else 1.0)
        now, then, yday = (level.get((sid, cutoff)), level.get((sid, yday_cut)),
                           level.get((sid, t - pd.Timedelta(days=1))))
        if None not in (now, then, yday) and then > 0:
            value = (1 - peso) * value + peso * yday * float(np.clip(now / then, limite_min, limite_max))
            seasonal_used += 1
        corrected.append(value)
    return np.round(np.maximum(np.array(corrected, dtype=float), 0)), {
        'ratios': {k: round(v, 4) for k, v in sorted(ratios.items())}, 'sin_datos': sorted(fallback),
        'estacional_usada': seasonal_used}


METHODS = {'modelo_base': modelo_base, 'ratio_ultima_hora': ratio_ultima_hora,
           'ratio_conjunto': ratio_conjunto, 'ratio_estacional': ratio_estacional}
MODEL_METHODS = {'modelo_sombra'}  # versiones que predicen con otro modelo registrado
BASE_VERSION = '1.0'  # version_sombra con metodo modelo_base: lo que se envía si algo falla


def lookback(params):
    """Observaciones de historial que necesita una versión (0 = ninguna)."""
    return max(int(params.get('ventana_obs') or 0), int(params.get('historial_obs') or 0),
               max((int(k) for k in params.get('ventanas') or ()), default=0))


def load_history(store, cutoff, window):
    """Observaciones (cutoff - window*15min, cutoff]; nunca posteriores al corte."""
    history = pd.DataFrame(columns=['station_id', 'observed_at', 'demand'])
    if window:
        rows = store.observations_window(cutoff - window * STEP, cutoff)
        if rows:
            history = pd.DataFrame(rows)
            history['observed_at'] = pd.to_datetime(history.observed_at, utc=True)
            history = history[history.observed_at <= cutoff]  # doble barrera contra fuga
    return history


def base_predictions(store, model, targets, cutoff):
    """Predicción del modelo activo. Los modelos con historia (lookback_obs > 0) reciben
    las observaciones hasta el corte; los de calendario predicen como siempre."""
    lookback = int(getattr(model, 'lookback_obs', 0) or 0)
    if lookback:
        history = load_history(store, cutoff, lookback)
        return np.asarray(model.predict_at(targets[['station_id', 'target_at']], history, cutoff), dtype=float)
    return np.asarray(model.predict(targets.rename(columns={'target_at': 'observed_at'})), dtype=float)


def delivery_values(store, base, targets, model, cutoff, version=None):
    """Valores a ENVIAR según version_envio (o la versión indicada). Nunca lanza: ante
    cualquier fallo devuelve el modelo base redondeado (idéntico al envío previo al selector)."""
    fallback = np.round(np.asarray(base, dtype=float))
    try:
        v = store.shadow_version(version) if version else store.delivery_version()
        if v['metodo'] not in METHODS:
            raise ValueError(f"metodo desconocido {v['metodo']}")
        history = load_history(store, cutoff, lookback(v['parametros']))
        values, diag = METHODS[v["metodo"]](np.asarray(base, dtype=float).copy(), targets, history, model, cutoff, **v['parametros'])
        values = np.asarray(values, dtype=float)
        if values.shape != fallback.shape or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('valores inválidos')
        return values, {'version': v['version'], **diag}
    except Exception as exc:
        return fallback, {'version': BASE_VERSION, 'fallback': type(exc).__name__}


def run_shadow(store, cycle):
    """Calcula las versiones en sombra que falten para este ciclo. Idempotente."""
    versions = [v for v in store.shadow_versions() if v['estado'] == 'sombra']
    done = store.shadow_done(cycle['cycle_id'])
    pending = [v for v in versions if v['version'] not in done]
    unknown = sorted(v['version'] for v in pending if v['metodo'] not in METHODS and v['metodo'] not in MODEL_METHODS)
    pending = [v for v in pending if v['metodo'] in METHODS or v['metodo'] in MODEL_METHODS]
    if not pending:
        return {'status': 'up_to_date', 'unknown_method': unknown}
    record = store.active_model()
    model = store.load_model(record)
    cutoff = timestamp(cycle['data_cutoff'])
    targets = pd.DataFrame(cycle['targets'])[['station_id', 'target_at']]
    base = base_predictions(store, model, targets, cutoff)
    history = load_history(store, cutoff, max(lookback(v['parametros']) for v in pending))
    computed = []
    for v in pending:
        sha = record['sha256']
        if v['metodo'] in MODEL_METHODS:
            # Otro modelo registrado (p. ej. la combinación sombra): predice igual que el activo.
            shadow = store.shadow_model(v['parametros']['nombre'])
            if shadow is None:
                continue
            values = base_predictions(store, store.load_model(shadow), targets, cutoff)
            diag, sha = {'modelo': shadow['version']}, shadow['sha256']
        else:
            values, diag = METHODS[v['metodo']](base, targets, history, model, cutoff, **v['parametros'])
        if not np.isfinite(values).all():
            raise ValueError(f"Versión {v['version']} produjo valores no finitos")
        store.save_shadow({
            'ciclo_id': cycle['cycle_id'], 'version': v['version'],
            'modelo_sha256': sha, 'codigo': os.getenv('GITHUB_SHA'),
            'data_cutoff': cycle['data_cutoff'], 'diagnostico': diag,
            'predicciones': [{'station_id': s, 'target_at': t, 'value': float(x)}
                             for s, t, x in zip(targets.station_id, targets.target_at, values)]})
        computed.append(v['version'])
    return {'status': 'computed', 'versions': computed, 'unknown_method': unknown}
