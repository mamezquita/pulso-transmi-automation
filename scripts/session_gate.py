"""Cerrar automáticamente la campaña al cumplir su fecha límite UTC."""
from datetime import datetime, timezone
import math
import os


def allowed_minutes(until, requested, now):
    end = datetime.fromisoformat(until.replace('Z', '+00:00'))
    if end.tzinfo is None:
        raise ValueError('AUTOMATION_UNTIL debe incluir zona horaria')
    requested = int(requested)
    if not 1 <= requested <= 300:
        raise ValueError('Duración permitida: 1..300 minutos')
    # Nunca exceder la fecha final; el margen inferior a un minuto no se ejecuta.
    return max(0, min(requested, math.floor((end - now).total_seconds() / 60)))


if __name__ == '__main__':
    minutes = allowed_minutes(os.environ['AUTOMATION_UNTIL'],
                              os.environ.get('REQUESTED_MINUTES', '300'),
                              datetime.now(timezone.utc))
    with open(os.environ['GITHUB_OUTPUT'], 'a') as out:
        out.write(f'minutes={minutes}\nenabled={str(minutes > 0).lower()}\n')
    print(f'Minutos autorizados para esta sesión: {minutes}')
