"""Sincroniza versiones, modelo base y evaluaciones de drift de Supabase a MLflow.

Almacén de MLflow: esquema `mlflow` de la misma base Supabase (no expuesto por la
API REST). Idempotente: se puede ejecutar cada hora.

Entorno (separado del operativo, para no alterar versiones de librerías):
    python -m venv ~/.venvs/pulso-mlflow && ~/.venvs/pulso-mlflow/bin/pip install mlflow psycopg2-binary
(la ruta del entorno no debe contener "%": Alembic, que usa MLflow, falla con ese carácter)
Variables: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY y MLFLOW_TRACKING_URI
(o MLFLOW_DB_PASSWORD, contraseña del rol mlflow_app, para construirla con el
proyecto tomado de SUPABASE_URL). En GitHub corre cada hora (workflow `MLflow`).
Uso:       python tools/mlflow_sync.py
Interfaz:  mlflow ui --backend-store-uri "$MLFLOW_TRACKING_URI"
"""
import json
import os
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime

os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')
import mlflow  # noqa: E402
from mlflow.entities import Metric  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

EXP_VERSIONES = 'pulso-transmi-versiones'
EXP_DRIFT = 'pulso-transmi-drift'
EXP_REENTRENOS = 'pulso-transmi-reentrenos'


def load_env(path=Path('.env')):
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def tracking_uri():
    """Rol propio mlflow_app: su search_path por defecto es el esquema mlflow, así
    que no depende de opciones de conexión (el pooler de Supabase las ignora)."""
    if os.getenv('MLFLOW_TRACKING_URI'):
        return os.environ['MLFLOW_TRACKING_URI']
    ref = os.getenv('SUPABASE_PROJECT_REF') or urllib.parse.urlparse(os.environ['SUPABASE_URL']).hostname.split('.')[0]
    pw = os.environ.get('MLFLOW_DB_PASSWORD') or Path('~/.config/proyecto1-supabase/mlflow-db-password').expanduser().read_text().strip()
    host = os.getenv('SUPABASE_POOLER_HOST', 'aws-0-us-east-1.pooler.supabase.com')
    return (f'postgresql+psycopg2://mlflow_app.{ref}:{urllib.parse.quote(pw, safe="")}@{host}:5432/postgres'
            '?sslmode=require')


def rows(table, **params):
    base = os.environ['SUPABASE_URL'].rstrip('/')
    key = os.environ['SUPABASE_SERVICE_ROLE_KEY']
    url = f'{base}/rest/v1/{table}?{urllib.parse.urlencode(params)}'
    req = urllib.request.Request(url, headers={'apikey': key, 'Authorization': f'Bearer {key}'})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def clean(name):
    """MLflow solo acepta ciertos caracteres en nombres de métricas."""
    return re.sub(r'[^A-Za-z0-9_\-. /]', '_', name.replace('%', 'pct'))


def experiment(client, name):
    exp = client.get_experiment_by_name(name)
    return exp.experiment_id if exp else client.create_experiment(name, artifact_location='mlflow-artifacts')


def find_run(client, exp_id, key, value):
    found = client.search_runs([exp_id], f"tags.{key} = '{value}'", max_results=1)
    return found[0] if found else None


def sync_versions(client):
    exp = experiment(client, EXP_VERSIONES)
    step = int(time.time() // 3600)  # un punto por hora de sincronización
    now_ms = int(time.time() * 1000)
    active = rows('version_envio', select='version', nombre='eq.demanda')[0]['version']
    verdict = {r['version']: r for r in rows('veredicto_sombra')}
    vs20 = {r['version_b']: r for r in rows('comparacion_versiones', version_a='eq.2.0')}
    logged = 0
    for v in rows('version_sombra', order='version.asc'):
        run = find_run(client, exp, 'version', v['version'])
        if run is None:
            run = client.create_run(exp, run_name=f"v{v['version']}", tags={
                'version': v['version'], 'metodo': v['metodo'], 'version_base': v['version_base'] or '',
                'mlflow.note.content': v['descripcion']})
            params = {'metodo': v['metodo'], 'version_base': v['version_base'] or '-'}
            params.update({k: json.dumps(x) if isinstance(x, (list, dict)) else x for k, x in v['parametros'].items()})
            for k, x in params.items():
                client.log_param(run.info.run_id, k, x)
        rid = run.info.run_id
        client.set_tag(rid, 'estado', v['estado'])
        client.set_tag(rid, 'activa', str(v['version'] == active).lower())
        metrics = {}
        ver = verdict.get(v['version'], {})
        for k in ('mae', 'mae_entregado', 'mejora_pct', 'ciclos_ganados_pct', 'ciclos_en_vivo'):
            if ver.get(k) is not None:
                metrics[f'vivo_{k}'] = ver[k]
        comp = vs20.get(v['version'], {})
        for k in ('diferencia_media', 'ic95_min', 'ic95_max', 'ciclos'):
            if comp.get(k) is not None:
                metrics[f'vs_2_0_{k}'] = comp[k]
        if metrics:  # un solo lote por versión: una ida y vuelta a la base
            client.log_batch(rid, metrics=[Metric(clean(k), float(x), now_ms, step) for k, x in metrics.items()])
        logged += len(metrics)
    return logged


def sync_base_model(client):
    exp = experiment(client, EXP_VERSIONES)
    active_sha = rows('modelo_activo', nombre='eq.demanda')[0]['modelo_sha256']
    for m in rows('registro_modelo_operativo', select='version,sha256,metadata'):
        if find_run(client, exp, 'modelo_sha256', m['sha256']):
            continue
        meta = m['metadata'] or {}
        run = client.create_run(exp, run_name=f"modelo-base {m['version']}", tags={
            'modelo_sha256': m['sha256'], 'tipo': 'modelo_base', 'activo': str(m['sha256'] == active_sha).lower()})
        rid = run.info.run_id
        for k in ('kind', 'training_data_end', 'trained_at', 'training_rows', 'python'):
            if meta.get(k) is not None:
                client.log_param(rid, k, meta[k])
        for pkg, ver in (meta.get('versions') or {}).items():
            client.log_param(rid, f'lib_{pkg}', ver)
        now_ms = int(time.time() * 1000)
        vals = [Metric(clean(f'validacion_{name}_{k}'), float(x), now_ms, 0)
                for name, scores in (meta.get('validation') or {}).get('scores', {}).items()
                for k, x in scores.items() if isinstance(x, (int, float))]
        if vals:
            client.log_batch(rid, metrics=vals)
        client.set_terminated(rid)


def sync_drift(client):
    exp = experiment(client, EXP_DRIFT)
    run = find_run(client, exp, 'tipo', 'monitoreo') or client.create_run(
        exp, run_name='monitoreo-drift', tags={'tipo': 'monitoreo', 'ultima_evaluacion': '0',
                                               'mlflow.note.content': 'Una serie por métrica; paso = evaluación de drift.'})
    rid = run.info.run_id
    last = int(client.get_run(rid).data.tags.get('ultima_evaluacion', '0'))
    decisions = rows('decision_drift', evaluacion_id=f'gt.{last}', order='evaluacion_id.asc')
    if not decisions:
        return 0
    ids = ','.join(str(d['evaluacion_id']) for d in decisions)
    per_eval = {}
    for m in rows('metrica_drift', evaluacion_id=f'in.({ids})'):
        per_eval.setdefault(m['evaluacion_id'], []).append(m)
    codes = {'conservar': 0, 'datos_insuficientes': 1, 'reentrenar': 2, 'revisar_datos': 3}
    for d in decisions:
        eid = d['evaluacion_id']
        ts = int(datetime.fromisoformat(d['evaluado_en']).timestamp() * 1000)
        batch = {'decision_codigo': codes[d['decision']]}
        for k in ('accuracy_acumulada', 'accuracy_24h', 'caida_accuracy_pp', 'ciclos_24h', 'nivel_vs_modelo_24h',
                  'cuarentena_pct_24h', 'intentos_degradados_24h', 'faltantes_pct_24h'):
            if d.get(k) is not None:
                batch[k] = d[k]
        for m in per_eval.get(eid, []):
            h = 'total' if m['horizonte_minutos'] == 0 else f"h{m['horizonte_minutos']}"
            for k in ('accuracy_macro', 'mae', 'rmse', 'sesgo'):
                if m.get(k) is not None:
                    batch[f"{m['ventana']}_{h}_{k}"] = m[k]
        # Un lote por evaluación; el marcador avanza solo si el lote se guardó.
        client.log_batch(rid, metrics=[Metric(clean(k), float(x), ts, eid) for k, x in batch.items()])
        client.set_tag(rid, 'ultima_evaluacion', str(eid))
        client.set_tag(rid, 'ultima_decision', d['decision'])
        client.set_tag(rid, 'ultimo_motivo', d['motivo'][:500])
    return len(decisions)


def sync_model_versions(client):
    """Una corrida por versión de modelo (Modelo N revisión R) en el experimento de versiones."""
    exp = experiment(client, EXP_VERSIONES)
    active = rows('modelo_activo_detalle', select='version_mayor,revision')
    active = (active[0]['version_mayor'], active[0]['revision']) if active else None
    fams = {f['version_mayor']: f['familia'] for f in rows('familia_modelo')}
    n = 0
    for v in rows('version_modelo', order='version_mayor.asc,revision.asc'):
        key = f"{v['version_mayor']}.r{v['revision']}"
        run = find_run(client, exp, 'modelo_version', key)
        if run is None:
            run = client.create_run(exp, run_name=f"modelo {v['version_mayor']} r{v['revision']} ({fams.get(v['version_mayor'], '?')})",
                                    tags={'modelo_version': key, 'tipo': 'version_modelo', 'mlflow.note.content': v['motivo']})
            for k, x in {**v['config'], 'modelo_sha256': v['modelo_sha256'][:16]}.items():
                client.log_param(run.info.run_id, k, json.dumps(x) if isinstance(x, (list, dict)) else x)
            vals = [Metric(clean(k), float(x), int(time.time() * 1000), 0) for k, x in (v['metricas'] or {}).items()
                    if isinstance(x, (int, float))]
            if vals:
                client.log_batch(run.info.run_id, metrics=vals)
            n += 1
        client.set_tag(run.info.run_id, 'activo', str(active == (v['version_mayor'], v['revision'])).lower())
    return n


def sync_retrainings(client):
    """Una corrida por candidato de cada reentreno: la tabla comparativa dentro de MLflow."""
    exp = experiment(client, EXP_REENTRENOS)
    n = 0
    for r in rows('reentreno', order='reentreno_id.asc'):
        for c in rows('candidato_reentreno', reentreno_id=f"eq.{r['reentreno_id']}"):
            key = f"{r['reentreno_id']}:{c['familia']}"
            if find_run(client, exp, 'candidato', key):
                continue
            run = client.create_run(exp, run_name=f"reentreno {r['reentreno_id']} · {c['familia']}", tags={
                'candidato': key, 'reentreno_id': str(r['reentreno_id']), 'familia': c['familia'],
                'elegido': str(c['elegido']).lower(), 'decision': r['decision'] or '', 'mlflow.note.content': r['motivo'] or ''})
            rid = run.info.run_id
            for k, x in c['config'].items():
                client.log_param(rid, k, json.dumps(x) if isinstance(x, (list, dict)) else x)
            ts = int(datetime.fromisoformat(r['iniciado_en']).timestamp() * 1000)
            vals = {'accuracy_prueba': c['accuracy'], 'accuracy_validacion': c['cv_accuracy'], 'mae': c['mae'],
                    'mape': c['mape'], 'wape_pct': 100 * c['wape'], 'pruebas': c['pruebas'], 'segundos': c['segundos']}
            if r.get('accuracy_actual') is not None:
                vals['accuracy_entregado_antes'] = r['accuracy_actual']
            client.log_batch(rid, metrics=[Metric(clean(k), float(x), ts, 0) for k, x in vals.items() if x is not None])
            client.set_terminated(rid)
            n += 1
    return n


def main():
    load_env()
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    sync_base_model(client)
    metrics = sync_versions(client)
    models = sync_model_versions(client)
    candidates = sync_retrainings(client)
    evaluations = sync_drift(client)
    print(json.dumps({'status': 'ok', 'metricas_versiones': metrics, 'versiones_modelo_nuevas': models,
                      'candidatos_nuevos': candidates, 'evaluaciones_drift_nuevas': evaluations}))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Los logs del repo público son visibles: solo el tipo de error, nunca la conexión.
        print(json.dumps({'status': 'error', 'error_type': type(exc).__name__}))
        raise SystemExit(1) from None
