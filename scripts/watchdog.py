"""Vigilante independiente: detecta sesiones caídas y horas sin entrega aceptada.

También revisa la salud de los datos (observaciones atrasadas, modelo atrasado,
cuarentena persistente, cambios de formato de la API) y abre un issue por alerta
en el repositorio; GitHub avisa al dueño por correo/app. El issue se cierra solo
cuando la condición se resuelve (los de formato los cierra el dueño).

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
DATA_LAG = timedelta(minutes=60)     # observaciones guardadas detrás del corte del último ciclo
MODEL_LAG = timedelta(hours=3)       # el refresco horario debería mantenerlo por debajo
QUARANTINE_ATTEMPTS = 4              # intentos seguidos con filas ilegibles
FORMAT_WINDOW = timedelta(hours=24)  # cambios de formato recientes que merecen issue
MISSING_REASON = 'Valor faltante reportado por la API'  # dato faltante declarado: no es un error
PREFIX = '[Alerta] '


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


def health(now, latest_cutoff, latest_obs, model_end, collectors, changes):
    """{clave: (título, cuerpo)} de las alertas activas. Funciones puras: fáciles de probar."""
    alerts = {}
    if latest_cutoff is not None and (latest_obs is None or latest_cutoff - latest_obs > DATA_LAG):
        lag = 'sin observaciones' if latest_obs is None else f'{(latest_cutoff - latest_obs).total_seconds() / 3600:.1f} h'
        alerts['datos'] = ('Datos atrasados', f'La última observación guardada ({latest_obs and latest_obs.isoformat()}) '
                           f'está {lag} detrás del corte del último ciclo ({latest_cutoff.isoformat()}). '
                           'Revisar la cuarentena del colector y la tabla `cambios_formato_api`.')
    if latest_cutoff is not None and model_end is not None and latest_cutoff - model_end > MODEL_LAG:
        alerts['modelo'] = ('Modelo atrasado', f'El modelo activo tiene datos hasta {model_end.isoformat()}, '
                            f'{(latest_cutoff - model_end).total_seconds() / 3600:.1f} h antes del corte '
                            f'({latest_cutoff.isoformat()}). Revisar el workflow Reentreno (refresco horario).')
    bad = [sum(1 for q in (c or {}).get('quarantine') or [] if q.get('reason') != MISSING_REASON) for c in collectors]
    if len(bad) >= QUARANTINE_ATTEMPTS and all(n > 0 for n in bad[:QUARANTINE_ATTEMPTS]):
        reasons = sorted({q.get('reason') for q in (collectors[0] or {}).get('quarantine') or []
                          if q.get('reason') != MISSING_REASON})
        alerts['cuarentena'] = ('Cuarentena persistente', f'{QUARANTINE_ATTEMPTS} intentos seguidos con filas '
                                f'ilegibles (último: {bad[0]}). Motivos: {", ".join(reasons)}.')
    for ch in changes:
        if now - parse(ch['primera_vez']) <= FORMAT_WINDOW:
            fields = ', '.join(f'`{k}`' for k in sorted(ch.get('cambios') or {}))
            alerts[f"formato-{ch['formato_id']}"] = (
                f"Cambio de formato en {ch['fuente']} (#{ch['formato_id']})",
                f"Detectado {ch['primera_vez']}. Campos: {fields}.\n\nDetalle (antes → ahora):\n```json\n"
                f"{json.dumps(ch.get('cambios'), ensure_ascii=False, indent=1)[:3000]}\n```\n"
                'Las entregas se adaptan solas; confirmar que la cuarentena y la ingesta siguen bien. '
                'Cerrar este issue a mano cuando esté revisado.')
    return alerts


def gh(method, path, body=None):
    url = f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        'Authorization': f"Bearer {os.environ['GITHUB_TOKEN']}", 'Accept': 'application/vnd.github+json'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def sync_issues(alerts, api=gh):
    """Abre un issue por alerta nueva y cierra los de alertas resueltas (salvo formato)."""
    open_issues = {i['title']: i for i in api('GET', '/issues?state=open&per_page=100')
                   if i['title'].startswith(PREFIX) and 'pull_request' not in i}
    wanted = {PREFIX + title: body for title, body in alerts.values()}
    opened, closed = [], []
    for title, body in wanted.items():
        if title not in open_issues:
            api('POST', '/issues', {'title': title, 'body': body}); opened.append(title)
    for title, issue in open_issues.items():
        if title not in wanted and not title.startswith(PREFIX + 'Cambio de formato'):
            api('POST', f"/issues/{issue['number']}/comments", {'body': 'Resuelto: la condición ya no se cumple.'})
            api('PATCH', f"/issues/{issue['number']}", {'state': 'closed'}); closed.append(title)
    return opened, closed


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


def check_health(now):
    cycles = query('ejecucion_operativa', select='data_cutoff', order='data_cutoff.desc', limit=1)
    obs = query('observaciones_disponibles', select='observado_en', order='observado_en.desc', limit=1)
    active = query('modelo_activo', select='modelo_sha256', nombre='eq.demanda')
    model = query('registro_modelo_operativo', select='metadata->>training_data_end',
                  sha256='eq.' + active[0]['modelo_sha256']) if active else []
    attempts = query('intento_operativo', select='collector:resultado->collector', order='iniciado_en.desc',
                     limit=QUARANTINE_ATTEMPTS, resultado='not.is.null')
    changes = query('cambios_formato_api', select='formato_id,fuente,primera_vez,cambios',
                    order='formato_id.desc', limit=10)
    end = model[0].get('training_data_end') if model else None
    alerts = health(now, parse(cycles[0]['data_cutoff']) if cycles else None,
                    parse(obs[0]['observado_en']) if obs else None, parse(end) if end else None,
                    [a.get('collector') for a in attempts], changes)
    if os.getenv('GITHUB_TOKEN') and os.getenv('GITHUB_REPOSITORY'):
        sync_issues(alerts)
    return alerts


def main():
    now = datetime.now(timezone.utc)
    until = parse(os.environ['AUTOMATION_UNTIL'])
    attempts = query('intento_operativo', select='iniciado_en', order='iniciado_en.desc', limit=1)
    deliveries = query('ejecucion_operativa', select='respuesta', respuesta='not.is.null',
                       order='actualizado_en.desc', limit=5)
    last_attempt = parse(attempts[0]['iniciado_en']) if attempts else None
    last_receipt = latest_receipt(deliveries)
    status = diagnose(now, until, last_attempt, last_receipt)
    report = {'status': status, 'checked_at': now.isoformat(),
              'last_attempt': last_attempt and last_attempt.isoformat(),
              'last_accepted_at': last_receipt and last_receipt.isoformat()}
    if status != 'campaign_over':
        try:
            report['alertas'] = sorted(check_health(now))
        except Exception as exc:  # la salud de datos nunca impide relanzar sesiones
            report['alertas_error'] = type(exc).__name__
    print(json.dumps(report))
    with open(os.environ.get('GITHUB_OUTPUT', os.devnull), 'a') as out:
        out.write(f'status={status}\n')


if __name__ == '__main__':
    main()
