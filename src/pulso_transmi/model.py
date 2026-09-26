"""Mezcla 50/50 LightGBM Poisson + perfil multiplicativo, serializable en joblib."""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TZ = 'America/Bogota'
ESTACIONES_EVENTO = ['Movistar Arena', 'Museo Nacional']
CLAVES_PERFIL = ['station_id', 'fin_de_semana', 'franja']
VARS_MULT = ['rain_mm', 'event_intensity', 'evento_estacion']
VARS_LGBM = ['estacion_cat', 'franja', 'fin_de_semana', 'rain_mm', 'event_intensity']
PARAMS_LGBM = dict(objective='poisson', n_estimators=3000, learning_rate=0.02,
    num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.9, reg_lambda=5, random_state=42, verbose=-1,
    n_jobs=4, deterministic=True, force_col_wise=True)


def fechas_locales(values: pd.Series) -> pd.Series:
    """Rechaza fechas ambiguas sin zona horaria; normaliza a Bogotá."""
    if values.isna().any():
        raise ValueError('observed_at contiene fechas vacías')
    for value in values.unique():
        if pd.Timestamp(value).tzinfo is None:
            raise ValueError('observed_at debe incluir zona horaria')
    result = pd.to_datetime(values, utc=True).dt.tz_convert(TZ)
    if ((result.dt.minute % 15 != 0) | (result.dt.second != 0)
            | (result.dt.microsecond != 0) | (result.dt.nanosecond != 0)).any():
        raise ValueError('observed_at debe estar alineado a intervalos de 15 minutos')
    return result


def evaluar(real, pred, estaciones=None) -> dict:
    real = np.asarray(real, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 0, None)
    if not len(real) or real.shape != pred.shape or not np.isfinite(real).all() or not np.isfinite(pred).all():
        raise ValueError('Métricas requieren arrays no vacíos, finitos y de igual tamaño')
    if (real < 0).any():
        raise ValueError('Demanda real negativa')
    error = np.abs(real-pred)
    positive = real > 0
    result = {'n': len(real), 'MAE': float(mean_absolute_error(real,pred)),
        'RMSE': float(np.sqrt(mean_squared_error(real,pred))),
        'MAPE_%': float(np.mean(error[positive]/real[positive])*100) if positive.any() else None,
        'MAPE_n': int(positive.sum()), 'R2': float(r2_score(real,pred)) if len(real)>1 else None}
    if estaciones is not None:
        sums = pd.DataFrame({'station_id': np.asarray(estaciones), 'real':real, 'error':error}).groupby('station_id')[['real','error']].sum()
        ratios = sums.loc[sums.real>0,'error']/sums.loc[sums.real>0,'real']
        result['WAPE_macro'] = float(ratios.mean()) if len(ratios) else None
        result['Accuracy_macro_%'] = float((100*(1-ratios).clip(lower=0)).mean()) if len(ratios) else None
        result['estaciones_evaluadas'] = len(ratios)
    return result


class ModeloDemanda:
    """Incluye estimadores, perfiles, catálogo, categorías y metadatos de entrenamiento."""
    def __init__(self, estaciones: pd.DataFrame, params: dict | None = None):
        self.estaciones = estaciones[['station_id','station_name']].copy()
        self.estaciones['station_id'] = self.estaciones.station_id.astype('string')
        if self.estaciones.isna().any().any() or self.estaciones.station_id.duplicated().any():
            raise ValueError('Catálogo incompleto o con estaciones duplicadas')
        self.categorias = sorted(self.estaciones.station_id.tolist())
        self.params = {**PARAMS_LGBM, **(params or {})}
        self.metadata = {}

    def crear_variables(self, datos: pd.DataFrame) -> pd.DataFrame:
        required = {'station_id','observed_at','event_intensity'}
        if required-set(datos):
            raise ValueError(f'Faltan columnas: {sorted(required-set(datos))}')
        d = datos.drop(columns=['perfil','station_name'], errors='ignore').copy().reset_index(drop=True)
        d['station_id'] = d.station_id.astype('string')
        if d.station_id.isna().any() or not d.station_id.isin(self.categorias).all():
            raise ValueError('Estación desconocida o vacía; se requiere reentrenar con su histórico')
        d['observed_at'] = fechas_locales(d.observed_at)
        # Fallback fila a fila; si no existe ninguna fuente válida se rechaza.
        rain = pd.to_numeric(d.get('rain_mm', pd.Series(np.nan,index=d.index)), errors='raise')
        if 'rain_forecast' in d:
            rain = rain.fillna(pd.to_numeric(d.rain_forecast,errors='raise'))
        d['rain_mm'] = rain
        d['event_intensity'] = pd.to_numeric(d.event_intensity,errors='raise')
        if not np.isfinite(d[['rain_mm','event_intensity']].to_numpy(dtype=float)).all():
            raise ValueError('Lluvia o intensidad de evento faltante/no finita')
        if (d.rain_mm<0).any() or not d.event_intensity.between(0,1).all():
            raise ValueError('Lluvia debe ser >=0 y event_intensity debe estar entre 0 y 1')
        d['station_name'] = d.station_id.map(self.estaciones.set_index('station_id').station_name)
        d['franja'] = d.observed_at.dt.hour*4+d.observed_at.dt.minute//15
        d['fin_de_semana'] = (d.observed_at.dt.dayofweek>=5).astype(int)
        d['dia_semana'] = d.observed_at.dt.dayofweek
        d['estacion_cat'] = pd.Categorical(d.station_id,categories=self.categorias)
        d['evento_estacion'] = d.event_intensity*d.station_name.isin(ESTACIONES_EVENTO)
        return d

    def fit(self, datos: pd.DataFrame) -> 'ModeloDemanda':
        d = self.crear_variables(datos)
        if d.empty or 'demand' not in d:
            raise ValueError('Entrenamiento requiere demanda e histórico no vacío')
        d['demand'] = pd.to_numeric(d.demand,errors='raise')
        if not np.isfinite(d.demand).all() or (d.demand<=0).any():
            raise ValueError('El modelo log-multiplicativo requiere demand > 0')
        if d.duplicated(['station_id','observed_at']).any():
            raise ValueError('Observaciones duplicadas')
        if set(d.station_id)!=set(self.categorias):
            raise ValueError('Cada estación del catálogo necesita histórico')
        self.perfil = d.groupby(CLAVES_PERFIL).demand.mean().rename('perfil')
        self.perfil_estacion = d.groupby('station_id').demand.mean()
        d = d.join(self.perfil,on=CLAVES_PERFIL)
        self.multiplicativo = LinearRegression().fit(d[VARS_MULT],np.log(d.demand)-np.log(d.perfil))
        self.lgbm = lgb.LGBMRegressor(**self.params).fit(d[VARS_LGBM],d.demand)
        self.metadata.update({'filas':len(d),'inicio':d.observed_at.min().isoformat(),
            'fin':d.observed_at.max().isoformat(),'timezone':TZ,'mezcla':[0.5,0.5],
            'params_lgbm':self.params,'features_lgbm':VARS_LGBM,'features_mult':VARS_MULT})
        return self

    def predict_components(self, nuevos: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self,'perfil'):
            raise ValueError('Modelo no entrenado')
        # La etiqueta nunca participa en el preprocesamiento ni en predicción.
        d = self.crear_variables(nuevos.drop(columns=['demand'],errors='ignore'))
        if d.empty:
            return d.assign(perfil=pd.Series(dtype=float),pred_mult=pd.Series(dtype=float),pred_lgbm=pd.Series(dtype=float),pred_mezcla=pd.Series(dtype=float))
        d = d.join(self.perfil,on=CLAVES_PERFIL)
        d['perfil'] = d.perfil.fillna(d.station_id.map(self.perfil_estacion))
        d['pred_mult'] = d.perfil.to_numpy()*np.exp(self.multiplicativo.predict(d[VARS_MULT]))
        d['pred_lgbm'] = self.lgbm.predict(d[VARS_LGBM])
        d['pred_mezcla'] = np.clip((d.pred_mult+d.pred_lgbm)/2,0,None)
        if not np.isfinite(d[['pred_mult','pred_lgbm','pred_mezcla']]).all().all():
            raise ValueError('Predicción no finita; revisar entradas fuera de distribución')
        return d

    def predict(self, nuevos: pd.DataFrame) -> pd.DataFrame:
        d = self.predict_components(nuevos)
        d['demanda_predicha'] = d.pred_mezcla.round().astype('int64')
        return d[['observed_at','station_id','station_name','demanda_predicha']]


def entrenar_modelo_final(datos_historicos, estaciones):
    return ModeloDemanda(estaciones).fit(datos_historicos)


def predecir(modelos, nuevos):
    return modelos.predict(nuevos)
