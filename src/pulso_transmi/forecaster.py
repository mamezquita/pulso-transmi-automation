"""Inferencia con modelo activo remoto y outbox persistido, sin reentrenar."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import uuid

import httpx
import numpy as np
import pandas as pd

from .client import DEFAULT_BASE_URL
from .contract import normalize_receipt
from .operational import load_env,normalize_cycle
from .persistence import RemoteStore
from .prepare_submission import check_current_cycle,timestamp,validate_payload
from .shadow import base_predictions,delivery_values


# Si el modelo activo falla (carga, validación o predicción), el ciclo sale con el modelo
# de calendario inicial + la corrección que se usó con él en vivo, en vez de no enviarse.
FALLBACK_MODEL=(1,0)
FALLBACK_CORRECTION='2.3'


def predict_base(store,record,cycle,raw_targets):
    meta=record['metadata']
    cutoff=timestamp(cycle['data_cutoff'])
    if timestamp(meta['training_data_end'])>cutoff:raise ValueError('Modelo entrenado después del corte')
    latest=store.latest_observation(cycle['data_cutoff'])
    if latest is None or timestamp(latest)<timestamp(meta['training_data_end']):
        raise ValueError('El histórico persistido no cubre el entrenamiento del modelo activo')
    model=store.load_model(record)
    if timestamp(model.end)!=timestamp(meta['training_data_end']):raise ValueError('Metadatos no coinciden con el artefacto')
    base=base_predictions(store,model,raw_targets,cutoff)
    if base.shape!=(len(raw_targets),) or not np.isfinite(base).all():raise ValueError('Predicciones no finitas')
    return model,base


def prepare(store,cycle):
    job=store.job(cycle['cycle_id'])
    if job:return job
    record=store.active_model()
    cutoff=timestamp(cycle['data_cutoff'])
    targets=pd.DataFrame(cycle['targets']).rename(columns={'target_at':'observed_at'})
    raw_targets=pd.DataFrame(cycle['targets'])[['station_id','target_at']]
    fallback=False
    try:
        model,base=predict_base(store,record,cycle,raw_targets)
        # Versión elegida en version_envio; ante cualquier fallo, modelo base (sin cambios).
        values,selection=delivery_values(store,base,raw_targets,model,cutoff)
    except Exception as exc:
        failed=record
        try:
            record=store.fallback_model(*FALLBACK_MODEL)
            if record['sha256']==failed['sha256']:raise exc  # el respaldo es el mismo modelo que falló
            model,base=predict_base(store,record,cycle,raw_targets)
        except Exception as fallback_exc:
            raise exc from fallback_exc  # se reporta la falla del modelo activo
        values,selection=delivery_values(store,base,raw_targets,model,cutoff,version=FALLBACK_CORRECTION)
        selection={**selection,'respaldo':{'modelo_fallido':failed['version'],'error':type(exc).__name__,'detalle':str(exc)[:300]}}
        fallback=True
    meta=record['metadata']
    payload={'schema_version':'1.0','cycle_id':cycle['cycle_id'],'client_run_id':str(uuid.uuid4()),
        'data_cutoff':cycle['data_cutoff'],'model':{'version':record['version'],
        'trained_at':meta['trained_at'],'training_data_end':meta['training_data_end']},
        'predictions':[{'station_id':row.station_id,'target_at':row.observed_at,'value':float(value)}
                       for row,value in zip(targets.itertuples(),values)]}
    validate_payload(payload,normalize_cycle(cycle))
    # Ganador de la carrera fija el payload; nunca enviar el payload candidato sin reservar.
    job=store.reserve(payload,record['sha256'],fallback=True) if fallback else store.reserve(payload,record['sha256'])
    if job.get('client_run_id')==payload['client_run_id']:
        try:store.record_delivery_version(cycle['cycle_id'],selection)
        except Exception:pass  # trazabilidad; nunca bloquea el envío
    return job


def forecast(store,api,*,submit=False,token=None,cycle=None):
    cycle=cycle or check_current_cycle(DEFAULT_BASE_URL,api)
    if cycle is None:return {'status':'no_open_cycle','submitted':False}
    job=prepare(store,cycle)
    if job['respuesta']:
        return {'status':'already_accepted','submitted':False,'cycle_id':job['ciclo_id'],
                'receipt':job['respuesta']}
    payload=job['payload'];validate_payload(payload,normalize_cycle(cycle))
    if not submit:return {'status':'prepared','submitted':False,'cycle_id':job['ciclo_id'],
                          'predictions':len(payload['predictions']),'client_run_id':job['client_run_id']}
    if not token:raise ValueError('Falta PULSO_API_KEY')
    live=check_current_cycle(DEFAULT_BASE_URL,api)
    if live is None or live['cycle_id']!=job['ciclo_id'] or live['state']!='open':
        return {'status':'cycle_closed_or_changed','submitted':False}
    # closes_at ilegible (drift de formato): no se inventa; decide la respuesta del servidor.
    if live.get('closes_at') and timestamp(datetime.now(timezone.utc))>=timestamp(live['closes_at']):
        return {'status':'cycle_closed','submitted':False}
    validate_payload(payload,normalize_cycle(live))
    try:
        r=api.post('/v1/submissions',json=payload,headers={'Authorization':f'Bearer {token}','Idempotency-Key':job['client_run_id']})
    except httpx.TransportError as exc:
        store.finish(job,error={'type':type(exc).__name__,'message':'Respuesta desconocida; reintentar el mismo payload y clave'})
        raise
    try:body=r.json()
    except ValueError:body={'message':'non_json_response'}
    if r.is_error:
        store.finish(job,error={'http_status':r.status_code,'response':body})
        r.raise_for_status()
    receipt={'http_status':r.status_code,'response':normalize_receipt(body,r.status_code)}
    # Si falla este guardado después del POST, el próximo runner reutiliza payload y clave.
    store.finish(job,response=receipt)
    return {'status':'accepted','submitted':True,'receipt':receipt}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file',type=Path,default=Path('.env'))
    parser.add_argument('--submit',action='store_true')
    parser.add_argument('--recover-cycle',help='Consultar una entrega persistida por ID, sin llamar a la API del profesor')
    args=parser.parse_args();load_env(args.env_file);store=RemoteStore()
    try:
        if args.recover_cycle:
            job=store.job(args.recover_cycle)
            print(json.dumps({'found':job is not None,'cycle_id':args.recover_cycle,
                'client_run_id':job['client_run_id'] if job else None,'receipt':job['respuesta'] if job else None}))
        else:
            with httpx.Client(base_url=DEFAULT_BASE_URL,timeout=60) as api:
                print(json.dumps(forecast(store,api,submit=args.submit,token=os.getenv('PULSO_API_KEY'))))
    finally:store.close()

if __name__=='__main__':main()
