"""Supabase REST/RPC + Storage privado; no estado local ni sesión de CLI necesarios."""
from __future__ import annotations
import hashlib
import importlib.metadata
import io
import os
from urllib.parse import urlparse

import httpx
import joblib

RUNTIME_PACKAGES=('lightgbm','scikit-learn','pandas','numpy','joblib','scipy')

class RemoteStore:
    def __init__(self,url=None,key=None,client=None):
        url=url or os.environ.get('SUPABASE_URL','')
        key=key or os.environ.get('SUPABASE_SERVICE_ROLE_KEY','')
        if not url or not key:raise ValueError('Faltan SUPABASE_URL y SUPABASE_SERVICE_ROLE_KEY')
        if urlparse(url).scheme!='https':raise ValueError('Supabase requiere HTTPS')
        self.client=client or httpx.Client(base_url=url.rstrip('/'),timeout=90,
            headers={'apikey':key,'Authorization':f'Bearer {key}'})

    def close(self):self.client.close()

    def request(self,method,path,**kwargs):
        r=self.client.request(method,path,**kwargs)
        if r.is_error:
            try:message=r.json().get('message',r.json().get('error','request_failed'))
            except ValueError:message='request_failed'
            raise RuntimeError(f'Supabase {r.status_code}: {message}')
        return r

    def rows(self,table,**params):
        return self.request('GET','/rest/v1/'+table,params=params).json()

    def rpc(self,name,**params):
        return self.request('POST','/rest/v1/rpc/'+name,json=params).json()

    def state(self):return self.rows('estado_colector',nombre='eq.observations')[0]

    def save_page(self,revision,rows,cursor):
        return self.rpc('ingerir_pagina',p_revision=revision,p_rows=rows,p_cursor=cursor)

    def active_model(self):
        active=self.rows('modelo_activo',nombre='eq.demanda')
        if not active:raise ValueError('No hay modelo activo registrado')
        return self.rows('registro_modelo_operativo',sha256='eq.'+active[0]['modelo_sha256'])[0]

    def load_model(self,record):
        expected=record['metadata']['versions']
        for package,version in expected.items():
            if importlib.metadata.version(package)!=version:
                raise ValueError(f'Versión incompatible: instalar {package}=={version}')
        content=self.request('GET','/storage/v1/object/authenticated/pulso-models/'+record['storage_path']).content
        if hashlib.sha256(content).hexdigest()!=record['sha256']:
            raise ValueError('El artefacto remoto no coincide con su SHA-256')
        return joblib.load(io.BytesIO(content))

    def job(self,cycle_id):
        result=self.rows('ejecucion_operativa',ciclo_id='eq.'+cycle_id)
        return result[0] if result else None

    def reserve(self,payload,sha,fallback=False):
        if not fallback:return self.rpc('reservar_submission',p_payload=payload,p_model_sha=sha)
        return self.rpc('reservar_submission',p_payload=payload,p_model_sha=sha,p_respaldo=True)

    def fallback_model(self,major,revision):
        version=self.rows('version_modelo',select='modelo_sha256',version_mayor=f'eq.{major}',revision=f'eq.{revision}')
        if not version:raise ValueError(f'No existe el modelo de respaldo {major} r{revision}')
        return self.rows('registro_modelo_operativo',sha256='eq.'+version[0]['modelo_sha256'])[0]

    def finish(self,job,response=None,error=None):
        return self.rpc('registrar_respuesta',p_cycle=job['ciclo_id'],p_run=job['client_run_id'],p_response=response,p_error=error)

    def latest_observation(self,cutoff):
        rows=self.rows('observaciones_disponibles',select='observado_en',observado_en='lte.'+cutoff,
                       order='observado_en.desc',limit='1')
        return rows[0]['observado_en'] if rows else None

    def start_attempt(self, attempt_id, submit):
        self.request('POST', '/rest/v1/intento_operativo', json={
            'intento_id': attempt_id, 'enviar': submit,
            'github_run_id': os.getenv('GITHUB_RUN_ID'),
            'github_run_attempt': os.getenv('GITHUB_RUN_ATTEMPT')})

    def finish_attempt(self, attempt_id, result):
        from datetime import datetime, timezone
        forecast = result.get('forecast', {})
        self.request('PATCH', '/rest/v1/intento_operativo',
            params={'intento_id': 'eq.' + attempt_id}, json={
                'terminado_en': datetime.now(timezone.utc).isoformat(),
                'estado': result.get('status', forecast.get('status', 'error')),
                'ciclo_id': result.get('cycle_id'),
                'registros_insertados': result.get('collector', {}).get('inserted'),
                'resultado': result})

    def observe_cycle(self, cycle):
        self.request('POST', '/rest/v1/ciclo_observado',
            params={'on_conflict': 'ciclo_id'},
            headers={'Prefer': 'resolution=ignore-duplicates'},
            json={'ciclo_id': cycle['cycle_id'], 'contrato': cycle})

    def shadow_versions(self):
        return self.rows('version_sombra', select='version,metodo,parametros,estado', order='version.asc')

    def shadow_done(self, cycle_id):
        rows = self.rows('pronostico_sombra', select='version', ciclo_id='eq.' + cycle_id)
        return {r['version'] for r in rows}

    def observations_window(self, start, end):
        # Dos filtros sobre la misma columna: (start, end]; nunca datos posteriores al corte.
        # PostgREST devuelve como máximo 1000 filas: paginar con orden estable.
        rows, offset = [], 0
        while True:
            page = self.request('GET', '/rest/v1/observaciones_disponibles', params=[
                ('select', 'estacion_id,observado_en,demanda'),
                ('observado_en', 'gt.' + start.isoformat()), ('observado_en', 'lte.' + end.isoformat()),
                ('order', 'observado_en.asc,estacion_id.asc'), ('limit', '1000'), ('offset', str(offset))]).json()
            rows += page
            offset += len(page)
            if len(page) < 1000:
                break
        return [{'station_id': r['estacion_id'], 'observed_at': r['observado_en'], 'demand': r['demanda']}
                for r in rows]

    def shadow_model(self, name):
        """Registro del modelo sombra `name` (None si aún no existe)."""
        rows = self.rows('modelo_sombra', select='modelo_sha256,familia,config', nombre='eq.' + name)
        if not rows:
            return None
        rec = self.rows('registro_modelo_operativo', sha256='eq.' + rows[0]['modelo_sha256'])[0]
        return {**rec, 'familia': rows[0]['familia'], 'config': rows[0]['config']}

    def set_shadow_model(self, row):
        self.request('POST', '/rest/v1/modelo_sombra', params={'on_conflict': 'nombre'},
                     headers={'Prefer': 'resolution=merge-duplicates'}, json=row)

    def save_shadow(self, row):
        self.request('POST', '/rest/v1/pronostico_sombra', params={'on_conflict': 'ciclo_id,version'},
                     headers={'Prefer': 'resolution=ignore-duplicates'}, json=row)

    def delivery_version(self):
        selected = self.rows('version_envio', select='version', nombre='eq.demanda')[0]['version']
        return self.shadow_version(selected)

    def shadow_version(self, version):
        return self.rows('version_sombra', select='version,metodo,parametros,estado', version='eq.' + version)[0]

    def record_delivery_version(self, cycle_id, selection):
        self.request('POST', '/rest/v1/envio_por_version', params={'on_conflict': 'ciclo_id'},
                     headers={'Prefer': 'resolution=ignore-duplicates'},
                     json={'ciclo_id': cycle_id, 'version': selection['version'], 'diagnostico': selection})

    def record_format(self, source, shape, sample):
        return self.rpc('registrar_formato', p_fuente=source, p_huella=shape, p_muestra=sample)

    def review_delivery_version(self):
        return self.rpc('revisar_version_envio')

    def evaluate_deliveries(self):
        return self.rpc('evaluar_entregas')
