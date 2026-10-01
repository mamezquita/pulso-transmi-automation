"""Prueba histórica de mezcla con el último valor y corrección de sesgo.

Solo lectura: no reserva, no envía ni cambia el modelo activo. Simula ciclos
horarios posteriores al entrenamiento usando únicamente datos <= data_cutoff.
Los parámetros se eligen en la primera mitad temporal y se miden en la segunda.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (15, 30, 45, 60)
STEP = pd.Timedelta(minutes=15)
WEIGHTS = np.round(np.arange(0, 1.0001, 0.05), 2)
WINDOWS = (1, 4, 12, 24, 96)  # observaciones de 15 min para estimar el sesgo
RATIO_LIMITS = (0.5, 2.0)


def origins_for(actual, start):
    """Orígenes horarios con valor en el corte y los 4 horizontes observados."""
    out = []
    for o in actual.index[(actual.index >= start) & (actual.index.minute == 0)]:
        needed = [o] + [o + pd.Timedelta(minutes=int(h)) for h in HORIZONS]
        if all(t in actual.index for t in needed) and actual.loc[needed].notna().all().all():
            out.append(o)
    return out


def bias_ratio(actual, model, origin, window):
    """Razón real/modelo por estación con las últimas `window` observaciones <= origen."""
    past_a = actual.loc[:origin].tail(window)
    past_m = model.loc[past_a.index]
    ratio = past_a.sum() / past_m.sum().replace(0, np.nan)
    return ratio.fillna(1.0).clip(*RATIO_LIMITS)


def simulate(actual, model, origins, *, window=None):
    """Filas por (origen, estación, horizonte) con real, modelo, último valor y modelo corregido."""
    rows = []
    for o in origins:
        last = actual.loc[o]
        ratio = bias_ratio(actual, model, o, window) if window else pd.Series(1.0, index=actual.columns)
        for h in HORIZONS:
            t = o + pd.Timedelta(minutes=int(h))
            for s in actual.columns:
                rows.append((o, s, h, actual.at[t, s], model.at[t, s], last[s], model.at[t, s] * ratio[s]))
    return pd.DataFrame(rows, columns=['origin', 'station_id', 'horizon', 'real', 'model', 'last', 'corrected'])


def blend(frame, weights, base='model'):
    w = frame.horizon.map(weights)
    return np.maximum(np.round(w * frame['last'] + (1 - w) * frame[base]), 0)


def metrics(real, pred):
    err = pred - real
    return {'n': int(len(real)), 'mae': float(np.abs(err).mean()),
            'rmse': float(np.sqrt((err ** 2).mean())),
            'wape_pct': float(100 * np.abs(err).sum() / real.sum()), 'bias': float(err.mean())}


def best_weights(frame, base):
    """Peso del último valor por horizonte que minimiza MAE en el tramo de ajuste."""
    chosen = {}
    for h, part in frame.groupby('horizon'):
        maes = [np.abs(np.round(w * part['last'] + (1 - w) * part[base]) - part.real).mean() for w in WEIGHTS]
        chosen[int(h)] = float(WEIGHTS[int(np.argmin(maes))])
    return chosen


def run(actual, model, start):
    origins = origins_for(actual, start)
    if len(origins) < 8:
        raise ValueError('Muy pocos ciclos posteriores al entrenamiento para separar ajuste y prueba')
    cut = origins[len(origins) // 2]
    base = simulate(actual, model, origins)
    tune, test = base[base.origin < cut], base[base.origin >= cut]

    # Elegir ventana de sesgo y pesos solo con el tramo de ajuste.
    by_window = {}
    for k in WINDOWS:
        f = simulate(actual, model, origins, window=k)
        f_tune = f[f.origin < cut]
        w = best_weights(f_tune, 'corrected')
        by_window[k] = (f, np.abs(np.round(f_tune.corrected) - f_tune.real).mean(),
                        np.abs(blend(f_tune, w, 'corrected') - f_tune.real).mean(), w)
    k_bias = min(by_window, key=lambda k: by_window[k][1])
    k_both = min(by_window, key=lambda k: by_window[k][2])
    w_model = best_weights(tune, 'model')
    w_both = by_window[k_both][3]
    test_bias = by_window[k_bias][0].pipe(lambda f: f[f.origin >= cut])
    test_both = by_window[k_both][0].pipe(lambda f: f[f.origin >= cut])

    variants = {
        'modelo_actual': (test, np.round(test.model)),
        'ultimo_valor': (test, test['last']),
        'mezcla': (test, blend(test, w_model)),
        'sesgo': (test_bias, np.round(test_bias.corrected)),
        'sesgo_y_mezcla': (test_both, blend(test_both, w_both, 'corrected')),
    }
    report = {'origins_total': len(origins), 'tune_until': cut.isoformat(),
              'test_origins': int(test.origin.nunique()),
              'params': {'mezcla_pesos': w_model, 'sesgo_ventana_obs': k_bias,
                         'sesgo_y_mezcla': {'ventana_obs': k_both, 'pesos': w_both}},
              'overall': {}, 'by_horizon': {}, 'by_station': {}, 'by_day': {}}
    for name, (frame, pred) in variants.items():
        report['overall'][name] = metrics(frame.real, pred)
        report['by_horizon'][name] = {int(h): metrics(frame.real[i], pred[i])['mae']
                                      for h, i in frame.groupby('horizon').groups.items()}
        report['by_station'][name] = {s: metrics(frame.real[i], pred[i])['mae']
                                      for s, i in frame.groupby('station_id').groups.items()}
        report['by_day'][name] = {str(d): metrics(frame.real[i], pred[i])['mae']
                                  for d, i in frame.groupby(frame.origin.dt.date).groups.items()}
    return report


def load_remote():
    from .operational import load_env
    from .persistence import RemoteStore
    load_env(Path('.env'))
    store = RemoteStore()
    try:
        record = store.active_model()
        model_obj = store.load_model(record)
        rows, offset = [], 0
        while True:
            page = store.rows('observaciones_disponibles', select='estacion_id,observado_en,demanda',
                              order='observado_en.asc,estacion_id.asc', limit=1000, offset=offset)
            rows += page
            offset += len(page)
            if len(page) < 1000:
                break
    finally:
        store.close()
    obs = pd.DataFrame(rows)
    obs['observado_en'] = pd.to_datetime(obs.observado_en, utc=True)
    actual = obs.pivot(index='observado_en', columns='estacion_id', values='demanda').astype(float)
    actual = actual.reindex(pd.date_range(actual.index.min(), actual.index.max(), freq=STEP))
    long = actual.stack(future_stack=True).rename('demand').reset_index()
    long.columns = ['observed_at', 'station_id', 'demand']
    long['model'] = model_obj.predict(long[['station_id', 'observed_at']])
    model = long.pivot(index='observed_at', columns='station_id', values='model')
    start = pd.Timestamp(record['metadata']['training_data_end']).tz_convert('UTC') + STEP
    return actual, model, start, record['version']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    actual, model, start, version = load_remote()
    report = run(actual, model, start)
    report.update(model_version=version, first_origin_after=start.isoformat(),
                  data_end=actual.dropna(how='all').index.max().isoformat(),
                  generated_at=datetime.now(timezone.utc).isoformat())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: report[k] for k in ('overall', 'params', 'test_origins')}, indent=2))


if __name__ == '__main__':
    main()
