"""Compatibilidad de inferencia del artefacto; no contiene histórico ni entrenamiento operativo."""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from .model import PARAMS_LGBM, fechas_locales
FEATURES=['station_cat','franja','fin_de_semana']
KEYS=['station_id','fin_de_semana','franja']

class CalendarModel:
    def __init__(self,kind='profile',params=None):
        self.kind=kind
        self.params={**PARAMS_LGBM,**(params or {})}

    def features(self,df):
        d=df[['station_id','observed_at']].copy().reset_index(drop=True)
        d['station_id']=d.station_id.astype('string')
        if not d.station_id.isin(self.categories).all():raise ValueError('Estación sin histórico')
        d['observed_at']=fechas_locales(d.observed_at)
        d['franja']=d.observed_at.dt.hour*4+d.observed_at.dt.minute//15
        d['fin_de_semana']=(d.observed_at.dt.dayofweek>=5).astype(int)
        d['station_cat']=pd.Categorical(d.station_id,categories=self.categories)
        return d

    def fit(self,df):
        if df.empty or not np.isfinite(df.demand).all() or (df.demand<0).any():raise ValueError('Histórico inválido')
        if df.duplicated(['station_id','observed_at']).any():raise ValueError('Duplicados en histórico')
        self.categories=sorted(df.station_id.astype(str).unique())
        d=self.features(df);d['demand']=df.demand.to_numpy()
        self.profile=d.groupby(KEYS).demand.mean().rename('profile')
        self.fallback=d.groupby('station_id').demand.mean()
        self.end=d.observed_at.max().isoformat()
        if self.kind=='lgbm':self.estimator=lgb.LGBMRegressor(**self.params).fit(d[FEATURES],d.demand)
        return self

    def predict(self,df):
        d=self.features(df)
        if self.kind=='lgbm':p=self.estimator.predict(d[FEATURES])
        else:
            d=d.join(self.profile,on=KEYS)
            p=d.profile.fillna(d.station_id.map(self.fallback)).to_numpy()
        if not np.isfinite(p).all():raise ValueError('Predicciones no finitas')
        return np.maximum(p,0)

def normalize_cycle(c):
    targets=c['targets']
    norm={'cycle_id':c['cycle_id'],'data_cutoff':c['data_cutoff'],
          'station_ids':sorted({r['station_id'] for r in targets}),
          'target_times':sorted({r['target_at'] for r in targets}),
          'expected_predictions':c['expected_predictions']}
    if len(targets)!=len({(r['station_id'],r['target_at']) for r in targets}):raise ValueError('Targets duplicados')
    return norm

def load_env(path):
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith('#'):
                key,sep,value=line.partition('=')
                if sep:os.environ.setdefault(key.strip(),value.strip().strip('\"\''))
