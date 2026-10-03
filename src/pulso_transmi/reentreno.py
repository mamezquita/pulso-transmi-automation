"""Etapa 4: reentreno exhaustivo y, si hace falta, cambio de modelo.

Política (tabla politica_reentreno): se dispara si el promedio de accuracy de los
últimos N ciclos baja del umbral de disparo. Primero se reentrena a fondo la familia
actual (LightGBM): hiperparámetros, conjuntos de variables y forma del objetivo, con
validación temporal. Si llega al umbral de cambio de modelo, se queda con esa familia
(nueva revisión). Si no, se prueban todas las demás familias con el mismo rigor y se
cambia solo si la mejor supera a LightGBM por el margen pactado (nueva versión).

Sin disparo, refresca el modelo activo: misma configuración con los datos más recientes
(nueva revisión). Con --sesion corre una pasada por hora sin depender del cron.

Nunca corre dentro de la sesión de envíos. Activa solo entre los minutos 10 y 30 de
una hora y después de confirmar la entrega de esa hora.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import backtest as bt
from . import busqueda as bq
from .client import DEFAULT_BASE_URL
from .prepare_submission import check_current_cycle
from .operational import load_env
from .persistence import RUNTIME_PACKAGES, RemoteStore
from .shadow import METHODS, base_predictions, lookback

LGBM = 'lightgbm'


def log(msg):
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


# ---------- lo que se entrega hoy, medido igual que los candidatos ----------
class _MatrixStore:
    """Sirve observaciones desde la matriz local, como lo haría Supabase."""
    def __init__(self, actual):
        self.actual = actual

    def observations_window(self, start, end):
        part = self.actual.loc[(self.actual.index > start) & (self.actual.index <= end)]
        return [{'station_id': s, 'observed_at': t, 'demand': v}
                for t, row in part.iterrows() for s, v in row.items() if not np.isnan(v)]


def current_score(actual, model, selector, test_origins):
    """Modelo activo + corrección activa sobre la prueba final (sin mirar el futuro)."""
    stations = sorted(actual.columns)
    reals, preds, sts = [], [], []
    for o in test_origins:
        tg = pd.DataFrame([{'station_id': s, 'target_at': (o + pd.Timedelta(minutes=h)).isoformat()}
                           for s in stations for h in (15, 30, 45, 60)])
        visible = _MatrixStore(actual.loc[:o])
        base = base_predictions(visible, model, tg, o)
        hist = pd.DataFrame(visible.observations_window(o - max(lookback(selector['parametros']), 1) * bt.STEP, o))
        if not hist.empty:
            hist['observed_at'] = pd.to_datetime(hist.observed_at, utc=True)
        else:
            hist = pd.DataFrame(columns=['station_id', 'observed_at', 'demand'])
        values, _ = METHODS[selector['metodo']](base, tg, hist, model, o, **selector['parametros'])
        for (s, t), v in zip(zip(tg.station_id, tg.target_at), values):
            r = actual.at[pd.Timestamp(t), s] if pd.Timestamp(t) in actual.index else np.nan
            if not np.isnan(r):
                reals.append(r); preds.append(v); sts.append(s)
    return bq.score(np.array(reals, float), np.round(np.array(preds, float)), np.array(sts))


def fair_current_score(actual, model, selector, test_cut, test_origins):
    """Lo entregado, medido igual que los candidatos. Si el modelo activo ya vio la prueba
    final al entrenarse, su puntaje estaría inflado: se reentrena su misma configuración
    solo con los datos anteriores a la prueba (los modelos con historia usan corrección 1.0)."""
    familia = getattr(model, 'familia', None)
    if familia and getattr(model, 'config', None) and pd.Timestamp(model.end) >= test_origins[0]:
        return {**bq.evaluate_config(test_cut, familia, model.config), 'reentrenado_sin_prueba': True}
    return current_score(actual, model, selector, test_origins)


# ---------- búsqueda según la política ----------
def decide(policy, current, lgbm, others):
    """Devuelve (decision, familia_elegida, resultado, fase, motivo)."""
    acc_l = lgbm['prueba']['accuracy']
    if acc_l >= policy['umbral_cambio_modelo']:
        if acc_l > current['accuracy']:
            return ('nueva_revision', LGBM, lgbm, 'arboles',
                    f"LightGBM ajustado llega a {acc_l:.1f} % (≥ {policy['umbral_cambio_modelo']} %) y supera lo entregado ({current['accuracy']:.1f} %)")
        return ('sin_cambio', None, None, 'arboles',
                f"LightGBM ajustado ({acc_l:.1f} %) no supera lo entregado ({current['accuracy']:.1f} %)")
    best = max(others, key=lambda r: r['prueba']['accuracy']) if others else None
    if best and best['prueba']['accuracy'] >= acc_l + policy['margen_cambio_pp'] and best['prueba']['accuracy'] > current['accuracy']:
        return ('nueva_version', best['familia'], best, 'otros_modelos',
                f"LightGBM ajustado se queda en {acc_l:.1f} % (< {policy['umbral_cambio_modelo']} %); "
                f"{best['familia']} logra {best['prueba']['accuracy']:.1f} % (+{best['prueba']['accuracy'] - acc_l:.1f} pp)")
    if acc_l > current['accuracy']:
        return ('nueva_revision', LGBM, lgbm, 'otros_modelos',
                f"Ningún otro modelo supera a LightGBM por {policy['margen_cambio_pp']} pp; LightGBM ajustado "
                f"({acc_l:.1f} %) mejora lo entregado ({current['accuracy']:.1f} %) aunque no llega a {policy['umbral_cambio_modelo']} %")
    return ('sin_cambio', None, None, 'otros_modelos', 'Ningún candidato mejora lo entregado; alerta: varianza alta')


# ---------- registro y activación ----------
def publish(store, model, version_mayor, revision, familia, result, motivo):
    buf = io.BytesIO(); joblib.dump(model, buf, compress=3)
    content = buf.getvalue(); digest = hashlib.sha256(content).hexdigest()
    path = f'{digest}/model.joblib'
    existing = store.client.get('/storage/v1/object/authenticated/pulso-models/' + path)
    if existing.status_code in (400, 404):
        store.request('POST', '/storage/v1/object/pulso-models/' + path, content=content,
                      headers={'Content-Type': 'application/octet-stream', 'x-upsert': 'false'})
    elif existing.status_code != 200 or hashlib.sha256(existing.content).hexdigest() != digest:
        raise ValueError('Artefacto remoto inconsistente')
    version = f'm{version_mayor}-{familia}-r{revision}'
    metadata = {'training_data_end': model.end, 'trained_at': datetime.now(timezone.utc).isoformat(),
                'training_rows': model.n_train, 'kind': 'historia', 'familia': familia,
                'versions': {k: importlib.metadata.version(k) for k in RUNTIME_PACKAGES},
                'python': platform.python_version(), 'config': result['config'],
                'validation': {'cv_accuracy': result['cv_accuracy'], 'pliegues': result['pliegues'], 'prueba': result['prueba']}}
    row = {'sha256': digest, 'version': version, 'storage_path': path, 'metadata': metadata}
    if not store.rows('registro_modelo_operativo', sha256='eq.' + digest):
        store.request('POST', '/rest/v1/registro_modelo_operativo', json=row)
    loaded = store.load_model(row)  # confirma descarga, hash y versiones antes de activar
    if loaded.end != model.end:
        raise ValueError('El artefacto descargado no coincide')
    store.request('POST', '/rest/v1/version_modelo', json={
        'version_mayor': version_mayor, 'revision': revision, 'modelo_sha256': digest,
        'config': {**result['config'], 'familia': familia, 'correccion': '1.0'},
        'metricas': {'cv_accuracy': result['cv_accuracy'], **result['prueba']}, 'motivo': motivo})
    return digest, version


def safe_now(store, now, open_cycle):
    """(seguro, motivo). Seguro = minutos 10–30, sesiones de envío con el código nuevo y
    entrega de esta hora confirmada (o ningún ciclo abierto)."""
    if not 10 <= now.minute < 30:
        return False, 'fuera de los minutos 10–30'
    # Solo el código que entiende modelos con historia registra las 4 etapas.
    attempt = store.rows('intento_operativo', select='resultado', order='iniciado_en.desc', limit=1)
    if not attempt or 'etapas' not in (attempt[0].get('resultado') or {}):
        return False, 'la sesión de envíos aún corre el código anterior (llega con el próximo relevo)'
    if open_cycle is None:
        return True, 'sin ciclo abierto'
    last = store.rows('ejecucion_operativa', select='ciclo_id,respuesta', respuesta='not.is.null',
                      order='actualizado_en.desc', limit=1)
    if bool(last) and last[0]['ciclo_id'] == open_cycle.get('cycle_id'):
        return True, 'entrega de esta hora confirmada'
    return False, 'hay un ciclo abierto sin entrega confirmada'


def wait_safe_window(store, max_wait):
    import httpx
    start, last_reason = time.time(), None
    while time.time() - start < max_wait:
        now = datetime.now(timezone.utc)
        if 10 <= now.minute < 30:
            with httpx.Client(timeout=30) as api:
                cycle = check_current_cycle(DEFAULT_BASE_URL, api)
            ok, reason = safe_now(store, now, cycle)
            if ok:
                return True
            if reason != last_reason:
                log(f'Aún no es seguro activar: {reason}')
                last_reason = reason
        time.sleep(60)
    return False


def activate(store, version_mayor, revision, motivo):
    # El modelo con historia ya incorpora el nivel reciente: corrección 1.0 (sin capa extra) y
    # respaldo 1.0 para que la reversión automática no vuelva a una corrección pensada para el
    # modelo de calendario.
    current = store.rows('version_envio', select='version,version_respaldo', nombre='eq.demanda')
    if not current or (current[0]['version'], current[0]['version_respaldo']) != ('1.0', '1.0'):
        store.request('PATCH', '/rest/v1/version_envio', params={'nombre': 'eq.demanda'},
                      json={'version': '1.0', 'version_respaldo': '1.0', 'motivo': f'Modelo {version_mayor} r{revision}: {motivo}'[:500]})
    return store.rpc('activar_modelo', p_version_mayor=version_mayor, p_revision=revision, p_motivo=motivo[:500])


def load_data():
    for attempt in range(3):
        try:
            actual, model_pivot, _, _ = bt.load_remote()
            origins = bt.origins_for(actual, actual.index[0] + pd.Timedelta(days=1, hours=4))
            return actual, origins
        except Exception as exc:
            log(f'Reintento de carga ({type(exc).__name__})'); time.sleep(20)
    raise RuntimeError('No se pudieron cargar los datos')


# ---------- refresco horario ----------
REFRESH_MIN_NEW_DATA = pd.Timedelta(minutes=30)  # el tiempo simulado avanza ~1 h por hora
REFRESH_KEEP_HOURS = 48


def refresh(store, dry_run=False, activation_wait=80 * 60):
    """Misma configuración del modelo activo, reentrenada con los datos más recientes.
    El ruido cambia rápido: un modelo con datos de 12 h atrás perdió 11 puntos en la
    prueba (2-oct). Queda como nueva revisión del mismo modelo, activada en ventana segura."""
    record = store.active_model()
    model = store.load_model(record)
    familia, cfg = getattr(model, 'familia', None), getattr(model, 'config', None)
    if not familia or not cfg:
        return {'status': 'sin_refresco', 'motivo': 'el modelo activo no tiene configuración para reentrenar'}
    actual, origins = load_data()
    latest = actual.dropna(how='all').index.max()
    if latest < pd.Timestamp(model.end) + REFRESH_MIN_NEW_DATA:
        return {'status': 'sin_refresco', 'motivo': 'el modelo activo ya tiene los datos recientes'}
    active = store.rows('modelo_activo_detalle', select='version_mayor,revision,metricas')[0]
    major, base_rev = active['version_mayor'], active['revision']
    motivo = f'Refresco horario de {major} r{base_rev}: misma configuración con datos hasta {latest.isoformat()}'
    log(motivo)
    if dry_run:
        final = bq.build_model(familia, cfg).fit(actual, origins)
        return {'status': 'simulado', 'motivo': motivo, 'end': final.end}
    rid = store.request('POST', '/rest/v1/reentreno', headers={'Prefer': 'return=representation'}, json={
        'disparo': {'motivo': 'refresco horario', 'modelo': f'{major} r{base_rev}'}, 'fase': 'refresco',
        'github_run_id': os.getenv('GITHUB_RUN_ID')}).json()[0]['reentreno_id']
    try:
        final = bq.build_model(familia, cfg).fit(actual, origins)
        if pd.Timestamp(final.end) <= pd.Timestamp(model.end):
            raise ValueError('El refresco no agregó datos nuevos')
        metricas = active.get('metricas') or {}
        result = {'config': {**cfg, 'origen': 'refresco', 'refresco_de': f'{major} r{base_rev}'},
                  'cv_accuracy': metricas.get('cv_accuracy'), 'pliegues': None,
                  'prueba': {k: v for k, v in metricas.items() if k != 'cv_accuracy'}}
        revs = store.rows('version_modelo', select='revision', version_mayor=f'eq.{major}', order='revision.desc', limit=1)
        rev = revs[0]['revision'] + 1
        digest, version = publish(store, final, major, rev, familia, result, motivo)
        log(f'Registrado {version} ({digest[:12]})')
        patch = {'estado': 'pendiente_activacion', 'decision': 'refresco', 'motivo': motivo,
                 'version_mayor': major, 'revision': rev}
        store.request('PATCH', '/rest/v1/reentreno', params={'reentreno_id': f'eq.{rid}'}, json=patch)
        if not wait_safe_window(store, max_wait=activation_wait):
            log('Sin ventana segura; el refresco queda pendiente para la próxima iteración')
            return {'status': 'pendiente_activacion', 'decision': 'refresco', 'motivo': motivo}
        log(f'Activado: {activate(store, major, rev, motivo)}')
        store.request('PATCH', '/rest/v1/reentreno', params={'reentreno_id': f'eq.{rid}'}, json={
            'estado': 'completado', 'terminado_en': datetime.now(timezone.utc).isoformat()})
    except Exception as exc:
        store.request('PATCH', '/rest/v1/reentreno', params={'reentreno_id': f'eq.{rid}'}, json={
            'terminado_en': datetime.now(timezone.utc).isoformat(), 'estado': 'error', 'error': str(exc)[:500]})
        raise
    try:
        log(f'Artefactos de refrescos antiguos borrados: {purge_old_refreshes(store)}')
    except Exception as exc:  # limpieza; nunca bloquea
        log(f'No se pudo limpiar artefactos antiguos ({type(exc).__name__})')
    return {'status': 'refrescado', 'decision': 'refresco', 'motivo': motivo, 'modelo': f'{major} r{rev}'}


def purge_old_refreshes(store, keep_hours=REFRESH_KEEP_HOURS, now=None):
    """Borra del storage los artefactos de refrescos con más de `keep_hours` que no estén
    activos (el registro y sus métricas se conservan; las búsquedas completas nunca se borran)."""
    now = now or datetime.now(timezone.utc)
    active = store.active_model()['sha256']
    cutoff = (now - pd.Timedelta(hours=keep_hours)).isoformat()
    old = store.rows('version_modelo', select='modelo_sha256', **{'config->>origen': 'eq.refresco', 'creada_en': f'lt.{cutoff}'})
    removed = 0
    for row in old:
        if row['modelo_sha256'] == active:
            continue
        reg = store.rows('registro_modelo_operativo', select='storage_path', sha256='eq.' + row['modelo_sha256'])
        if reg:
            r = store.client.delete('/storage/v1/object/pulso-models/' + reg[0]['storage_path'])
            removed += r.status_code == 200
    return removed


# ---------- orquestación ----------
def run(force=False, trials_lgbm=40, trials_otros=12, budget_family=1800, dry_run=False, activation_wait=4.5 * 3600,
        refrescar=True):
    load_env(Path('.env'))
    store = RemoteStore()
    started = time.time()
    pending = store.rows('reentreno', estado='eq.pendiente_activacion', order='reentreno_id.desc', limit=1)
    if pending and not dry_run:
        p = pending[0]
        log(f"Reentreno {p['reentreno_id']} pendiente: activar modelo {p['version_mayor']} r{p['revision']}")
        if not wait_safe_window(store, max_wait=activation_wait):
            log('Sigue pendiente; se reintentará en la próxima ejecución')
            store.close()
            return {'status': 'pendiente_activacion', 'reentreno_id': p['reentreno_id']}
        log(f"Activado: {activate(store, p['version_mayor'], p['revision'], p['motivo'])}")
        store.request('PATCH', '/rest/v1/reentreno', params={'reentreno_id': f"eq.{p['reentreno_id']}"},
                      json={'estado': 'completado', 'terminado_en': datetime.now(timezone.utc).isoformat()})
        store.close()
        return {'status': 'activado', 'reentreno_id': p['reentreno_id']}
    check = store.rpc('necesita_reentreno')
    log(f"Disparo: {check['disparar']} ({check['motivo']})")
    if not (check['disparar'] or force):
        if not refrescar or check['motivo'] == 'hay un modelo pendiente de activación':
            store.close()
            return {'status': 'sin_disparo', 'motivo': check['motivo']}
        try:
            return refresh(store, dry_run=dry_run, activation_wait=min(activation_wait, 80 * 60))
        finally:
            store.close()
    policy = check['politica']
    rid = None
    if not dry_run:
        rid = store.request('POST', '/rest/v1/reentreno', headers={'Prefer': 'return=representation'}, json={
            'disparo': check, 'github_run_id': os.getenv('GITHUB_RUN_ID')}).json()[0]['reentreno_id']
    try:
        actual, origins = load_data()
        cuts = bq.cortes(actual, origins)
        test = cuts[1].Xe.origin.drop_duplicates().tolist()
        active_model = store.load_model(store.active_model())
        current = fair_current_score(actual, active_model, store.delivery_version(), cuts[1], test)
        log(f"Entregado hoy en la prueba final: {current['accuracy']:.2f} %")

        lgbm, _ = bq.search_family(cuts, LGBM, trials=trials_lgbm, time_budget=budget_family, log=log)
        log(f"LightGBM ajustado: prueba {lgbm['prueba']['accuracy']:.2f} %")
        others = []
        if lgbm['prueba']['accuracy'] < policy['umbral_cambio_modelo']:
            for fam in [f for f in bq.FAMILIES if f != LGBM]:
                res, _ = bq.search_family(cuts, fam, trials=trials_otros, time_budget=budget_family, log=log)
                log(f"{fam}: prueba {res['prueba']['accuracy']:.2f} %")
                others.append(res)
            combo = bq.search_combination(cuts, [lgbm] + others, log=log)
            if combo:
                log(f"combinacion ({'+'.join(m['familia'] for m in combo['config']['miembros'])}): "
                    f"prueba {combo['prueba']['accuracy']:.2f} %")
                others.append(combo)
        decision, familia, chosen, fase, motivo = decide(policy, current, lgbm, others)
        log(f'Decisión: {decision} — {motivo}')
        table = [lgbm] + others
        if not dry_run:
            for r in table:
                store.request('POST', '/rest/v1/candidato_reentreno', json={
                    'reentreno_id': rid, 'familia': r['familia'], 'config': r['config'],
                    'cv_accuracy': r['cv_accuracy'], 'pliegues': r['pliegues'], 'accuracy': r['prueba']['accuracy'],
                    'wape': r['prueba']['wape'], 'mae': r['prueba']['mae'], 'mape': r['prueba']['mape'],
                    'predicciones': r['prueba']['n'], 'pruebas': r['pruebas_realizadas'], 'segundos': r['segundos'],
                    'elegido': r is chosen})
        version = None
        if decision != 'sin_cambio' and not dry_run:
            families = {f['familia']: f['version_mayor'] for f in store.rows('familia_modelo')}
            if familia not in families:
                major = max(families.values()) + 1
                store.request('POST', '/rest/v1/familia_modelo', json={
                    'familia': familia, 'version_mayor': major, 'descripcion': f'{familia} (elegido por reentreno {rid})'})
            else:
                major = families[familia]
            revs = store.rows('version_modelo', select='revision', version_mayor=f'eq.{major}', order='revision.desc', limit=1)
            rev = revs[0]['revision'] + 1 if revs else 0
            final = bq.build_model(familia, chosen['config'])
            final.fit(actual, origins)  # todos los datos, incluida la prueba final
            digest, version = publish(store, final, major, rev, familia, chosen, motivo)
            log(f'Registrado {version} ({digest[:12]}); esperando ventana segura para activar')
            version = (major, rev)
            store.request('PATCH', '/rest/v1/reentreno', params={'reentreno_id': f'eq.{rid}'}, json={
                'estado': 'pendiente_activacion', 'fase': fase, 'decision': decision, 'motivo': motivo,
                'accuracy_actual': current['accuracy'], 'accuracy_elegido': chosen['prueba']['accuracy'],
                'version_mayor': major, 'revision': rev})
            remaining = max(0, activation_wait - (time.time() - started))
            if not wait_safe_window(store, max_wait=remaining):
                log('Sin ventana segura dentro del tiempo del job; queda pendiente para la próxima ejecución')
                return {'status': 'pendiente_activacion', 'decision': decision, 'motivo': motivo, 'actual': current,
                        'tabla': [{'familia': r['familia'], **r['prueba'], 'cv': r['cv_accuracy']} for r in table]}
            log(f"Activado: {activate(store, major, rev, motivo)}")
        if not dry_run:
            store.request('PATCH', '/rest/v1/reentreno', params={'reentreno_id': f'eq.{rid}'}, json={
                'terminado_en': datetime.now(timezone.utc).isoformat(), 'estado': 'completado', 'fase': fase,
                'decision': decision, 'motivo': motivo, 'accuracy_actual': current['accuracy'],
                'accuracy_elegido': chosen['prueba']['accuracy'] if chosen else None,
                'version_mayor': version[0] if version else None, 'revision': version[1] if version else None})
        return {'status': 'completado', 'decision': decision, 'motivo': motivo, 'actual': current,
                'tabla': [{'familia': r['familia'], **r['prueba'], 'cv': r['cv_accuracy']} for r in table]}
    except Exception as exc:
        if rid is not None:
            store.request('PATCH', '/rest/v1/reentreno', params={'reentreno_id': f'eq.{rid}'}, json={
                'terminado_en': datetime.now(timezone.utc).isoformat(), 'estado': 'error', 'error': str(exc)[:500]})
        raise
    finally:
        store.close()


SESSION_MIN_REMAINING = 75 * 60  # una búsqueda completa + espera de ventana caben en este margen


def next_slot(now, minute=10):
    """Próximo minuto `minute` de una hora (inicio de la ventana segura)."""
    slot = now.replace(minute=minute, second=0, microsecond=0)
    return slot if slot > now else slot + pd.Timedelta(hours=1)


def session(minutes, sleep=time.sleep, clock=lambda: datetime.now(timezone.utc), **kw):
    """Una pasada por hora (activación pendiente, búsqueda completa o refresco) hasta agotar
    la sesión. El cron de GitHub se retrasa o se salta horas; esta sesión no depende de él."""
    deadline = clock() + pd.Timedelta(minutes=minutes)
    outs = []
    while (deadline - clock()).total_seconds() >= SESSION_MIN_REMAINING:
        remaining = (deadline - clock()).total_seconds()
        try:
            out = run(activation_wait=min(4.5 * 3600, remaining - 10 * 60), **kw)
        except Exception as exc:  # los logs son públicos: solo el tipo de error
            out = {'status': 'error', 'error_type': type(exc).__name__}
        log(f"Iteración: {out.get('status')} — {out.get('motivo', out.get('error_type', ''))}")
        outs.append(out)
        write_summary(out)
        wake = next_slot(clock())
        if (deadline - wake).total_seconds() < SESSION_MIN_REMAINING:
            break
        sleep((wake - clock()).total_seconds())
    return {'status': 'sesion_terminada', 'iteraciones': len(outs)}


def write_summary(out):
    summary = os.getenv('GITHUB_STEP_SUMMARY')
    if not summary:
        return
    with open(summary, 'a') as f:
        if out.get('tabla') and out.get('decision'):
            f.write(f"## Reentreno: {out['decision']}\n\n{out['motivo']}\n\n"
                    f"Entregado hoy: {out['actual']['accuracy']:.1f} %\n\n| Modelo | Accuracy | MAE | MAPE | WAPE |\n|---|---|---|---|---|\n")
            for r in sorted(out['tabla'], key=lambda r: -r['accuracy']):
                f.write(f"| {r['familia']} | {r['accuracy']:.1f} % | {r['mae']:.1f} | {r['mape']:.1f} % | {100 * r['wape']:.1f} % |\n")
        else:
            f.write(f"- {datetime.now(timezone.utc):%H:%M} {out.get('status')}: {out.get('motivo', out.get('error_type', ''))}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--forzar', action='store_true', help='Ignorar el disparo (pruebas manuales)')
    p.add_argument('--simular', action='store_true', help='No escribe en Supabase ni activa nada')
    p.add_argument('--pruebas-lightgbm', type=int, default=40)
    p.add_argument('--pruebas-otros', type=int, default=12)
    p.add_argument('--presupuesto-familia', type=int, default=1800, help='Segundos máximos por familia')
    p.add_argument('--sin-refresco', action='store_true', help='Sin disparo, no refrescar el modelo activo')
    p.add_argument('--sesion', type=int, default=0, help='Minutos de sesión con una pasada por hora (0 = una pasada)')
    a = p.parse_args()
    kw = dict(trials_lgbm=a.pruebas_lightgbm, trials_otros=a.pruebas_otros, budget_family=a.presupuesto_familia,
              dry_run=a.simular, refrescar=not a.sin_refresco)
    if a.sesion:
        print(json.dumps(session(a.sesion, **kw), ensure_ascii=False))
        return
    out = run(a.forzar, **kw)
    print(json.dumps(out, ensure_ascii=False, default=str))
    write_summary(out)


if __name__ == '__main__':
    main()
