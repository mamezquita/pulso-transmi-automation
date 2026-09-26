"""Vigilante independiente: detecta sesiones caídas y horas sin entrega aceptada.

Solo usa la biblioteca estándar y no imprime recibos ni datos personales.
Código de salida 1 = alerta (GitHub envía correo al dueño del workflow).
"""
from datetime import datetime, timedelta, timezone
import json
import os
import urllib.parse
import urllib.request

STALL = timedelta(minutes=20)       # sin intentos => la cadena de sesiones murió
NO_DELIVERY = timedelta(minutes=90)  # sin recibo => se perdió al menos una ventana


def parse(ts):
    return datetime.fromisoformat(ts.replace('Z', '+00:00'))


def diagnose(now, until, last_attempt, last_receipt):
    if now >= until:
        return 'campaign_over'
    if last_attempt is None or now - last_attempt > STALL:
        return 'stalled'
    if last_receipt is None or now - last_receipt > NO_DELIVERY:
        return 'no_recent_delivery'
    return 'ok'


def query(table, **params):
    base = os.environ['SUPABASE_URL'].rstrip('/')
    key = os.environ['SUPABASE_SERVICE_ROLE_KEY']
    url = f'{base}/rest/v1/{table}?{urllib.parse.urlencode(params)}'
    req = urllib.request.Request(url, headers={'apikey': key, 'Authorization': f'Bearer {key}'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def latest_receipt(rows):
    times = []
    for row in rows:
        response = (row.get('respuesta') or {}).get('response') or {}
        if response.get('submission_id') and response.get('received_at'):
            times.append(parse(response['received_at']))
    return max(times, default=None)


def main():
    now = datetime.now(timezone.utc)
    until = parse(os.environ['AUTOMATION_UNTIL'])
    attempts = query('intento_operativo', select='iniciado_en', order='iniciado_en.desc', limit=1)
    deliveries = query('ejecucion_operativa', select='respuesta', respuesta='not.is.null',
                       order='actualizado_en.desc', limit=5)
    last_attempt = parse(attempts[0]['iniciado_en']) if attempts else None
    last_receipt = latest_receipt(deliveries)
    status = diagnose(now, until, last_attempt, last_receipt)
    print(json.dumps({'status': status, 'checked_at': now.isoformat(),
                      'last_attempt': last_attempt and last_attempt.isoformat(),
                      'last_accepted_at': last_receipt and last_receipt.isoformat()}))
    with open(os.environ.get('GITHUB_OUTPUT', os.devnull), 'a') as out:
        out.write(f'status={status}\n')


if __name__ == '__main__':
    main()
