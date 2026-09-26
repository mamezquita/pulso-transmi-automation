"""Sesión acotada de vigilancia; el estado de cada intento permanece en Supabase."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import signal
import threading
import time

import httpx

from .automation import execute, save_report
from .client import DEFAULT_BASE_URL
from .operational import load_env
from .persistence import RemoteStore


def watch(attempt, emit, *, minutes, interval=120, clock=time.monotonic,
          sleep=time.sleep, stopped=lambda: False):
    if not 1 <= minutes <= 300:
        raise ValueError('La duración debe estar entre 1 y 300 minutos')
    if not 60 <= interval <= 300:
        raise ValueError('El intervalo debe estar entre 60 y 300 segundos')
    deadline = clock() + minutes * 60
    failures = 0
    attempts = 0
    accepted = 0
    while clock() < deadline and not stopped():
        try:
            result = attempt()
        except Exception as exc:
            result = {'status': 'error', 'error_type': type(exc).__name__}
        attempts += 1
        accepted += int(result.get('forecast', {}).get('submitted', False))
        failures = failures + 1 if result.get('status') == 'error' else 0
        emit(result)
        if failures >= 5:
            return {'status': 'failed', 'attempts': attempts, 'new_submissions': accepted}
        # Esperas interrumpibles; nunca iniciar otro intento después del plazo.
        next_attempt = min(deadline, clock() + interval)
        while clock() < next_attempt and not stopped():
            sleep(min(60, next_attempt - clock()))
    return {'status': 'interrupted' if stopped() else 'finished',
            'attempts': attempts, 'new_submissions': accepted}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minutes', type=int, required=True)
    parser.add_argument('--interval', type=int, default=120)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.minutes <= 300 or not 60 <= args.interval <= 300:
        parser.error('Duración 1..300 minutos; intervalo 60..300 segundos')
    load_env(Path('.env'))
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *unused: stop.set())
    started = datetime.now(timezone.utc)
    session = {'status': 'running', 'started_at': started.isoformat(),
               'planned_end': (started + timedelta(minutes=args.minutes)).isoformat(),
               'interval_seconds': args.interval}
    session_path = args.report.with_name('sesion.json')
    session_path.parent.mkdir(parents=True, exist_ok=True)
    session_path.write_text(json.dumps(session, indent=2))
    print(json.dumps(session), flush=True)

    def attempt():
        store = RemoteStore()
        try:
            with httpx.Client(base_url=DEFAULT_BASE_URL, timeout=60) as api:
                return execute(store, api, submit=args.submit, token=os.getenv('PULSO_API_KEY'))
        finally:
            store.close()

    result = watch(attempt, lambda report: save_report(report, args.report),
                   minutes=args.minutes, interval=args.interval,
                   sleep=stop.wait, stopped=stop.is_set)
    session.update(result, ended_at=datetime.now(timezone.utc).isoformat())
    session_path.write_text(json.dumps(session, indent=2))
    print(json.dumps(session), flush=True)
    if result['status'] != 'finished':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
