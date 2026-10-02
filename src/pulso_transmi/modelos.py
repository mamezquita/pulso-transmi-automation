"""Modelos de pronóstico con historia reciente (para reentreno y cambio de modelo).

`ModeloHistoria` pronostica cada objetivo (estación, hora objetivo) usando solo
información disponible al corte: calendario, perfil histórico y la demanda reciente
(último valor, medias, variabilidad, tendencia y el mismo momento del día anterior).
Así puede adaptarse a cambios de nivel y de varianza sin reentrenar cada hora.

El estimador es intercambiable (LightGBM, bosque aleatorio, lineal, KNN, red neuronal…)
y cada uno lleva su propio preprocesamiento (codificación de la estación, hora cíclica,
normalización). Solo usa scikit-learn y LightGBM, ya presentes en el entorno de envíos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STEP = pd.Timedelta(minutes=15)
DAY_STEPS = 96
TZ = 'America/Bogota'
HORIZONS = (15, 30, 45, 60)

FEATURE_SETS = {
    # Lo mismo que el modelo actual: solo calendario y perfil.
    'calendario': ['st', 'franja', 'dow', 'finde', 'h', 'perfil_t'],
    # Calendario + nivel y variabilidad recientes.
    'reciente': ['st', 'franja', 'dow', 'finde', 'h', 'perfil_t', 'perfil_o', 'ultimo', 'media1h',
                 'media3h', 'desv3h', 'pendiente', 'nivel'],
    # Lo anterior + el día anterior a la misma hora.
    'reciente_ayer': ['st', 'franja', 'dow', 'finde', 'h', 'perfil_t', 'perfil_o', 'ultimo', 'media1h',
                      'media3h', 'desv3h', 'pendiente', 'nivel', 'ayer_t', 'ayer_o', 'ayer_ratio'],
}
NEEDS_HISTORY = 4 * 3 + DAY_STEPS + 4  # 3 h de historia + un día + margen


def _local(times):
    return pd.DatetimeIndex(times).tz_convert(TZ)


class ModeloHistoria:
    """Interfaz que usa el envío: `end`, `lookback_obs`, `predict_at()` y `predict()`."""

    lookback_obs = NEEDS_HISTORY

    def __init__(self, estimador, familia, feature_set='reciente_ayer', objetivo='relativo', config=None):
        self.estimador = estimador
        self.familia = familia
        self.feature_set = feature_set
        self.objetivo = objetivo  # 'directo' o 'relativo' (demanda / referencia)
        self.config = config or {}

    # ---- perfil histórico (aprendido solo con datos de entrenamiento) ----
    def _fit_profile(self, actual):
        local = _local(actual.index)
        long = actual.assign(franja=local.hour * 4 + local.minute // 15, finde=(local.dayofweek >= 5).astype(int))
        long = long.melt(id_vars=['franja', 'finde'], var_name='station_id', value_name='demand').dropna()
        self.profile = long.groupby(['station_id', 'finde', 'franja']).demand.mean()
        self.station_mean = long.groupby('station_id').demand.mean()
        self.stations = sorted(self.station_mean.index)

    def _profile_at(self, stations, times):
        local = _local(times)
        keys = pd.MultiIndex.from_arrays([np.asarray(stations), (local.dayofweek >= 5).astype(int),
                                          local.hour * 4 + local.minute // 15])
        values = self.profile.reindex(keys).to_numpy()
        fallback = pd.Series(np.asarray(stations)).map(self.station_mean).to_numpy()
        return np.where(np.isnan(values), fallback, values)

    # ---- variables a partir de una matriz tiempo × estación ----
    def _features(self, actual, origins, stations, horizons):
        a = actual.reindex(columns=self.stations).to_numpy(float)
        pos = {t: i for i, t in enumerate(actual.index)}
        rows = []
        for o in origins:
            p = pos[o]
            win = lambda k: a[max(0, p - k + 1):p + 1]
            last, m1, m3 = a[p], np.nanmean(win(4), 0), np.nanmean(win(12), 0)
            sd3 = np.nanstd(win(12), 0)
            slope = a[p] - a[p - 4] if p >= 4 else np.full(len(self.stations), np.nan)
            yo = a[p - DAY_STEPS] if p >= DAY_STEPS else np.full(len(self.stations), np.nan)
            prof_o_win = np.vstack([self._profile_at(self.stations, [actual.index[q]] * len(self.stations))
                                    for q in range(max(0, p - 3), p + 1)])
            level = m1 / np.where(np.nanmean(prof_o_win, 0) > 0, np.nanmean(prof_o_win, 0), np.nan)
            prof_o = self._profile_at(self.stations, [o] * len(self.stations))
            for h in horizons:
                t = o + pd.Timedelta(minutes=int(h))
                q = p + int(h) // 15
                yt = a[q - DAY_STEPS] if q - DAY_STEPS >= 0 else np.full(len(self.stations), np.nan)
                prof_t = self._profile_at(self.stations, [t] * len(self.stations))
                lt = _local([t])[0]
                for j, s in enumerate(self.stations):
                    if s not in stations:
                        continue
                    rows.append((o, t, s, h, j, lt.hour * 4 + lt.minute // 15, lt.dayofweek, int(lt.dayofweek >= 5),
                                 prof_t[j], prof_o[j], last[j], m1[j], m3[j], sd3[j], slope[j], level[j],
                                 yt[j], yo[j], last[j] / yo[j] if yo[j] and yo[j] > 0 else np.nan))
        cols = ['origin', 'target_at', 'station_id', 'h', 'st', 'franja', 'dow', 'finde', 'perfil_t', 'perfil_o',
                'ultimo', 'media1h', 'media3h', 'desv3h', 'pendiente', 'nivel', 'ayer_t', 'ayer_o', 'ayer_ratio']
        return pd.DataFrame(rows, columns=cols)

    def _reference(self, X):
        ref = X['perfil_t'] * X['nivel'].clip(0.33, 3).fillna(1.0)
        return ref.where(ref > 1, X['perfil_t'].clip(lower=1))

    # ---- entrenamiento y predicción ----
    def prepare(self, actual, origins):
        """Perfil + variables de entrenamiento; reutilizable entre configuraciones."""
        self._fit_profile(actual)
        X = self._features(actual, origins, set(self.stations), HORIZONS)
        y = np.array([actual.at[t, s] for t, s in zip(X.target_at, X.station_id)], dtype=float)
        ok = ~np.isnan(y)
        self.end = actual.dropna(how='all').index.max().tz_convert(TZ).isoformat()
        return X[ok].reset_index(drop=True), y[ok]

    def adopt(self, other):
        """Copia el perfil aprendido por otro modelo (mismos datos de entrenamiento)."""
        for k in ('profile', 'station_mean', 'stations', 'end'):
            setattr(self, k, getattr(other, k))
        return self

    def fit_prepared(self, X, y):
        target = y / self._reference(X).to_numpy() if self.objetivo == 'relativo' else y
        self.estimador.fit(X[FEATURE_SETS[self.feature_set]], target)
        self.n_train = int(len(X))
        return self

    def fit(self, actual, origins):
        """`actual`: matriz tiempo × estación con datos hasta el último origen + 60 min."""
        X, y = self.prepare(actual, origins)
        return self.fit_prepared(X, y)

    def _predict_frame(self, X):
        raw = self.estimador.predict(X[FEATURE_SETS[self.feature_set]])
        out = raw * self._reference(X).to_numpy() if self.objetivo == 'relativo' else raw
        return np.maximum(np.asarray(out, dtype=float), 0)

    def predict_matrix(self, actual, origins):
        """Predicciones para varios orígenes (backtest): devuelve el DataFrame de variables + pred."""
        X = self._features(actual, origins, set(self.stations), HORIZONS)
        return X.assign(pred=self._predict_frame(X))

    def predict_at(self, targets, history, cutoff):
        """Envío: `targets` con station_id y target_at; `history` con observaciones <= cutoff."""
        cutoff = pd.Timestamp(cutoff).tz_convert('UTC')
        hist = history[pd.to_datetime(history.observed_at, utc=True) <= cutoff]
        idx = pd.date_range(cutoff - (self.lookback_obs - 1) * STEP, cutoff, freq=STEP)
        mat = (hist.assign(observed_at=pd.to_datetime(hist.observed_at, utc=True))
               .pivot_table(index='observed_at', columns='station_id', values='demand', aggfunc='last')
               .reindex(index=idx, columns=self.stations))
        # Extender con filas vacías hasta +60 min para indexar los objetivos (sin valores futuros).
        mat = mat.reindex(pd.date_range(idx[0], cutoff + pd.Timedelta(minutes=60), freq=STEP))
        targets = pd.DataFrame(targets)
        X = self._features(mat, [cutoff], set(targets.station_id), HORIZONS)
        X['target_at'] = pd.to_datetime(X.target_at, utc=True)
        pred = pd.Series(self._predict_frame(X), index=pd.MultiIndex.from_arrays([X.station_id, X.target_at]))
        keys = pd.MultiIndex.from_arrays([targets.station_id.astype(str), pd.to_datetime(targets.target_at, utc=True)])
        values = pred.reindex(keys).to_numpy()
        if np.isnan(values).any():
            raise ValueError('Objetivos fuera de los horizontes conocidos del modelo')
        return values

    def predict(self, df):
        """Compatibilidad con la corrección de sombra: perfil de calendario sin historia."""
        df = pd.DataFrame(df)
        return self._profile_at(df.station_id.astype(str), pd.to_datetime(df.observed_at, utc=True))


class ModeloCombinado:
    """Promedio de varios ModeloHistoria (familias distintas con errores distintos).
    Misma interfaz que usa el envío: `end`, `lookback_obs`, `predict_at()` y `predict()`."""
    familia = 'combinacion'

    def __init__(self, miembros, config=None):
        self.miembros = list(miembros)
        self.config = config or {}

    @property
    def lookback_obs(self):
        return max(m.lookback_obs for m in self.miembros)

    @property
    def end(self):
        return self.miembros[0].end

    @property
    def n_train(self):
        return self.miembros[0].n_train

    def fit(self, actual, origins):
        for m in self.miembros:
            m.fit(actual, origins)
        return self

    def predict_at(self, targets, history, cutoff):
        return np.mean([m.predict_at(targets, history, cutoff) for m in self.miembros], axis=0)

    def predict(self, df):
        return np.mean([np.asarray(m.predict(df), dtype=float) for m in self.miembros], axis=0)
