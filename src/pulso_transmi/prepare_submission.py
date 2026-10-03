"""Prepara y valida una submission local. Nunca ejecuta POST ni requiere API key."""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import re
import uuid

import httpx
import joblib
import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator, FormatChecker, ValidationError

from .client import DEFAULT_BASE_URL
from . import diagnostico
from .contract import normalize_cycle_contract


def timestamp(value):
    t = pd.Timestamp(value)
    if pd.isna(t) or t.tzinfo is None:
        raise ValueError('Los timestamps deben ser válidos y tener zona horaria')
    return t.tz_convert('UTC')


def validate_cycle(cycle):
    """Formato LOCAL explícito, no se asume que sea el formato de respuesta remota."""
    required = {'cycle_id','data_cutoff','station_ids','target_times'}
    if required-set(cycle):
        raise ValueError(f'Faltan campos del manifiesto local: {sorted(required-set(cycle))}')
    if not re.fullmatch(r'cyc_[A-Za-z0-9_-]{1,80}',cycle['cycle_id']):
        raise ValueError('cycle_id inválido')
    cutoff = timestamp(cycle['data_cutoff'])
    stations = cycle['station_ids']
    targets = [timestamp(t) for t in cycle['target_times']]
    if not stations or any(not isinstance(s,str) or not re.fullmatch(r'[0-9]{5}',s) for s in stations):
        raise ValueError('station_ids debe contener IDs de cinco dígitos como texto')
    if len(set(stations))!=len(stations) or len(set(targets))!=len(targets):
        raise ValueError('Estaciones u objetivos duplicados')
    if not targets or any(t<=cutoff or t.value % 900_000_000_000 for t in targets):
        raise ValueError('Objetivos deben ser posteriores al corte y estar alineados a 15 minutos')
    if len(stations)*len(targets)>100:
        raise ValueError('El contrato admite máximo 100 predicciones')
    if cycle.get('expected_predictions',len(stations)*len(targets))!=len(stations)*len(targets):
        raise ValueError('expected_predictions no coincide con estaciones × objetivos')
    return cutoff, {(s,t) for s in stations for t in targets}


def build_payload(model, cycle, context, *, model_version='mezcla-v1', run_id=None):
    cutoff, expected = validate_cycle(cycle)
    if timestamp(model.metadata['fin'])>cutoff:
        raise ValueError('El modelo fue entrenado con observaciones posteriores al corte')
    required = {'target_at','rain_forecast','event_intensity','known_at'}
    if required-set(context):
        raise ValueError(f'Contexto requiere {sorted(required)}')
    c = context.copy()
    c['target_at'] = c.target_at.map(timestamp)
    c['known_at'] = c.known_at.map(timestamp)
    if c.target_at.duplicated().any():
        raise ValueError('Contexto global duplicado para un objetivo')
    if set(c.target_at)!={t for _,t in expected}:
        raise ValueError('El contexto debe cubrir exactamente los objetivos del ciclo')
    if (c.known_at>cutoff).any():
        raise ValueError('Contexto publicado después del corte: riesgo de información futura')
    # Selección explícita: nunca usar rain_mm observada, demand ni columnas adicionales.
    grid = pd.DataFrame(sorted(expected),columns=['station_id','target_at'])
    inputs = grid.merge(c[['target_at','rain_forecast','event_intensity']],on='target_at',validate='many_to_one')
    inputs = inputs.rename(columns={'target_at':'observed_at'})
    predicted = model.predict(inputs)
    payload = {'schema_version':'1.0','cycle_id':cycle['cycle_id'],
        'client_run_id':run_id or str(uuid.uuid4()),'data_cutoff':cutoff.isoformat(),
        'model':{'version':model_version,'training_data_end':model.metadata['fin']},
        'predictions':[{'station_id':str(row.station_id),'target_at':row.observed_at.isoformat(),
                        'value':int(row.demanda_predicha)} for row in predicted.itertuples(index=False)]}
    if model.metadata.get('created_at'):
        payload['model']['trained_at'] = model.metadata['created_at']
    validate_payload(payload,cycle)
    return payload, predicted


def validate_payload(payload, cycle):
    schema = json.loads(files('pulso_transmi').joinpath('submission_schema.json').read_text())
    Draft202012Validator(schema,format_checker=FormatChecker()).validate(payload)
    cutoff,expected = validate_cycle(cycle)
    if payload['cycle_id']!=cycle['cycle_id'] or timestamp(payload['data_cutoff'])!=cutoff:
        raise ValueError('El payload no coincide con ciclo/corte')
    if timestamp(payload['model']['training_data_end'])>cutoff:
        raise ValueError('Modelo con datos futuros')
    actual = [(r['station_id'],timestamp(r['target_at'])) for r in payload['predictions']]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise ValueError('Cobertura incorrecta: faltan, sobran o se duplican predicciones')
    if not all(np.isfinite(r['value']) for r in payload['predictions']):
        raise ValueError('Predicciones no finitas')
    if not 8<=len(payload['client_run_id'])<=128:
        raise ValueError('client_run_id debe permitir Idempotency-Key (8 a 128 caracteres)')


def check_current_cycle(base_url, client):
    response = client.get(base_url.rstrip('/')+'/v1/forecast-cycles/current')
    if response.status_code==404 and response.json().get('detail',{}).get('code')=='no_open_cycle':
        return None
    response.raise_for_status()
    raw=response.json()
    diagnostico.capture('ciclo',raw)  # cruda, antes de normalizar: sirve aunque no se pueda leer
    # Formatos alternativos de fecha/tipos se traducen aquí; lo ambiguo se rechaza.
    return normalize_cycle_contract(raw)


def demo_inputs(model):
    cutoff = timestamp(model.metadata['fin'])
    # Horizontes SOLO de simulación, no atribuidos al ciclo oficial.
    targets = [(cutoff.to_pydatetime()+timedelta(minutes=m)).isoformat() for m in (15,30,45,60)]
    cycle = {'cycle_id':'cyc_demo_NO_ENVIAR','data_cutoff':cutoff.isoformat(),
        'station_ids':model.categorias,'target_times':targets,
        'expected_predictions':len(model.categorias)*len(targets)}
    context = pd.DataFrame({'target_at':targets,'rain_forecast':[0.,.5,1.,0.],
                            'event_intensity':[0.,0.,.3,.8],'known_at':[cutoff.isoformat()]*4})
    return cycle,context


def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--demo',action='store_true',help='Prueba offline, datos artificiales, NO ENVIAR')
    mode.add_argument('--cycle-file',type=Path,help='Manifiesto LOCAL con objetivos confirmados del ciclo')
    p.add_argument('--context',type=Path,help='CSV global por target: rain_forecast,event_intensity,known_at')
    p.add_argument('--model',type=Path,default=Path('models/mezcla_v1/modelo_final.joblib'))
    p.add_argument('--model-version',default='mezcla-v1')
    p.add_argument('--output-dir',type=Path,required=True,help='Carpeta nueva para esta ejecución')
    p.add_argument('--base-url',default=DEFAULT_BASE_URL)
    args=p.parse_args(argv)
    if args.context and not args.cycle_file:
        p.error('--context requiere --cycle-file')
    if args.cycle_file and not args.context:
        p.error('--cycle-file requiere --context')
    if args.output_dir.exists():
        p.error('La carpeta de salida ya existe; usar otra para no confundir envíos anteriores')
    try:
        if not args.demo and not args.cycle_file:
            with httpx.Client(timeout=30) as client:
                cycle=check_current_cycle(args.base_url,client)
            args.output_dir.mkdir(parents=True)
            if cycle is None:
                write_json(args.output_dir/'estado.json',{'status':'no_open_cycle','submitted':False})
                print('No hay ciclo abierto. No se generó submission ni se envió nada.')
            else:
                write_json(args.output_dir/'ciclo_api.json',cycle)
                write_json(args.output_dir/'estado.json',{'status':'cycle_available_requires_inputs','submitted':False})
                print('Ciclo guardado en ciclo_api.json. Confirmar objetivos y contexto; usar --cycle-file y --context. No se envió nada.')
            return 0
        model=joblib.load(args.model)
        if args.demo:
            cycle,context=demo_inputs(model)
        else:
            cycle=json.loads(args.cycle_file.read_text())
            if cycle['cycle_id'].startswith('cyc_demo'):
                raise ValueError('Un ciclo demo solo se procesa con --demo')
            context=pd.read_csv(args.context)
        payload,predicted=build_payload(model,cycle,context,model_version=args.model_version)
        args.output_dir.mkdir(parents=True)
        filename='submission.demo.NO_ENVIAR.json' if args.demo else 'submission.json'
        write_json(args.output_dir/filename,payload)
        write_json(args.output_dir/'ciclo_local.json',cycle)
        context.to_csv(args.output_dir/'contexto.csv',index=False)
        predicted.to_csv(args.output_dir/'predicciones.csv',index=False)
        digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
        report={'status':'validated_offline','demo':args.demo,'submitted':False,
            'server_acceptance_verified':False,'predictions':len(payload['predictions']),
            'cycle_id':cycle['cycle_id'],'client_run_id':payload['client_run_id'],
            'idempotency_key':payload['client_run_id'],'model_sha256':digest(args.model),
            'payload_sha256':digest(args.output_dir/filename),
            'note':'Contexto artificial, NO ENVIAR' if args.demo else 'Validación local; reconfirmar ventana y contrato antes de enviar'}
        write_json(args.output_dir/'validacion.json',report)
        print(f'{len(predicted)} predicciones validadas: {args.output_dir/filename}. No se envió nada.')
        return 0
    except (ValueError,KeyError,OSError,httpx.HTTPError,ValidationError) as exc:
        p.exit(2,f'No se preparó submission: {exc}\n')

if __name__=='__main__':
    raise SystemExit(main())
