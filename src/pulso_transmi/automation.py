"""Ejecución remota con identidad verificada y resultado auditable."""
import argparse
from datetime import datetime, timezone
import json
import os
import time
import uuid
from pathlib import Path

import httpx

from .client import DEFAULT_BASE_URL
from .collector import collect
from .forecaster import forecast
from .operational import load_env
from .persistence import RemoteStore
from . import diagnostico
from .contract import ContractError, canonical_keys
from .prepare_submission import check_current_cycle
from .shadow import run_shadow


class _Etapa:
    """Registra inicio, fin, duración y estado de una etapa en result['etapas']."""
    def __init__(self, result, name):
        self.result, self.name = result, name
    def __enter__(self):
        self.t0 = time.monotonic(); self.inicio = datetime.now(timezone.utc)
        self.row = {'etapa': self.name, 'inicio': self.inicio.isoformat(), 'estado': 'ok', 'resultado': {}}
        self.result.setdefault('etapas', []).append(self.row)
        return self.row
    def __exit__(self, exc_type, exc, tb):
        self.row['fin'] = datetime.now(timezone.utc).isoformat()
        self.row['ms'] = int((time.monotonic() - self.t0) * 1000)
        if exc_type is not None:
            self.row['estado'] = 'error'; self.row['resultado'] = {'error_type': exc_type.__name__}
        return False

EXPECTED_PARTICIPANT = 'stu_fbbcf1d8318b4f75990c2b582255bdba'

EXPECTED_PARTICIPANT = os.getenv('EXPECTED_PARTICIPANT_ID', '')


def execute(store, api, *, submit=False, token=None):
    result = {'started_at': datetime.now(timezone.utc).isoformat(), 'submit_requested': submit,
              'attempt_id': str(uuid.uuid4())}
    store.start_attempt(result['attempt_id'], submit)
    cycle = None
    try:
        if submit:
            if not token:
                raise ValueError('Falta PULSO_API_KEY')
            response = api.get('/v1/me', headers={'Authorization': f'Bearer {token}'})
            response.raise_for_status()
            identity = canonical_keys(response.json(), ('participant_id', 'display_name'))
            if str(identity.get('participant_id')).strip() != str(EXPECTED_PARTICIPANT):
                raise ValueError('La API key no pertenece al participante esperado')
            result['participant'] = identity.get('display_name')
        try:
            with _Etapa(result, 'colector') as et:
                result['collector'] = collect(store, api)
                et['resultado'] = {'insertados': result['collector'].get('inserted'),
                                   'cuarentena': result['collector'].get('quarantined', 0)}
        except Exception as exc:
            # La ingesta nunca bloquea el envío: el modelo de calendario no la necesita.
            result['collector'] = {'status': 'error', 'error_type': type(exc).__name__}
            if isinstance(exc, ContractError):
                result['collector']['detail'] = str(exc)[:200]
            result['degraded'] = True
        with _Etapa(result, 'envio') as et:
            cycle = check_current_cycle(DEFAULT_BASE_URL, api)
            if cycle is not None:
                result['cycle_id'] = cycle['cycle_id']
                store.observe_cycle(cycle)
                result['forecast'] = forecast(store, api, submit=submit, token=token, cycle=cycle)
            else:
                result['forecast'] = {'status': 'no_open_cycle', 'submitted': False}
            et['resultado'] = {'estado': result['forecast'].get('status'), 'enviado': result['forecast'].get('submitted', False)}
    except Exception as exc:
        result.update(status='error', error_type=type(exc).__name__)
    # Sombra después del envío: calcula versiones alternativas sin reservar ni enviar.
    if cycle is not None:
        try:
            result['shadow'] = run_shadow(store, cycle)
        except Exception as exc:
            result['shadow'] = {'status': 'error', 'error_type': type(exc).__name__}
    # Evaluación posterior al envío para no gastar la ventana en métricas.
    # También evaluar si falló el pronóstico, siempre que la ingesta terminó.
    if 'collector' in result:
        try:
            with _Etapa(result, 'drift') as et:
                result['evaluation'] = store.evaluate_deliveries()
                drift = result['evaluation'].get('drift') or {}
                et['resultado'] = {'estado': drift.get('status'), 'decision': drift.get('decision')}
        except Exception as exc:
            result.update(status='error', evaluation_error_type=type(exc).__name__)
    # Reversión automática de la versión enviada si su respaldo rinde mejor (12 ciclos).
    try:
        result['version_review'] = store.review_delivery_version()
    except Exception as exc:
        result['version_review'] = {'status': 'error', 'error_type': type(exc).__name__}
    # Diagnóstico de formato de la API (ciclo, stream, recibo): solo registra cambios.
    try:
        result['formato'] = diagnostico.flush(store)
    except Exception as exc:
        result['formato'] = {'status': 'error', 'error_type': type(exc).__name__}
    try:
        store.finish_attempt(result['attempt_id'], result)
    except Exception as exc:
        result.update(status='error', persistence_error_type=type(exc).__name__)
    return result


def save_report(result, path):
    # El historial detallado queda en Supabase, no en logs/artefactos públicos.
    forecast_result = result.get('forecast', {})
    safe = {'checked_at': datetime.now(timezone.utc).isoformat(),
            'status': result.get('status', forecast_result.get('status', 'error')),
            'submitted': forecast_result.get('submitted', False),
            'evaluation_complete': result.get('evaluation', {}).get('complete_cycles'),
            'evaluation_pending': result.get('evaluation', {}).get('pending_cycles'),
            'degraded': result.get('degraded', False),
            'quarantined': result.get('collector', {}).get('quarantined', 0),
            'shadow': result.get('shadow', {}).get('status'),
            'version_review': result.get('version_review', {}).get('status')}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe) + '\n')
    print(json.dumps(safe), flush=True)
    if os.getenv('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as stream:
            stream.write(f"- {safe['checked_at']}: {safe['status']}; envío nuevo={safe['submitted']}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    load_env(Path('.env'))
    store = None
    try:
        store = RemoteStore()
        with httpx.Client(base_url=DEFAULT_BASE_URL, timeout=60) as api:
            result = execute(store, api, submit=args.submit, token=os.getenv('PULSO_API_KEY'))
    except Exception as exc:
        # No incluir mensajes/headers que pudieran contener credenciales.
        save_report({'status': 'error', 'error_type': type(exc).__name__}, args.report)
        raise SystemExit(1) from None
    finally:
        if store is not None:
            store.close()
    save_report(result, args.report)
    if result.get('status') == 'error':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
