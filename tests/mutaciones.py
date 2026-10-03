"""Banco de mutaciones: los cambios de formato que el profesor puede introducir.

Cada mutación toma una respuesta real de la API (ciclo, página del stream o recibo)
y devuelve la misma información con otro formato: fechas como texto en otros
órdenes o como epoch, números como texto, booleanos como yes/no, estaciones sin
ceros, nombres de campos distintos. La información es la misma, así que el
resultado normalizado debe ser idéntico al del original.
"""
import copy
from datetime import datetime, timezone
from email.utils import format_datetime

import pandas as pd

CYCLE_DATES = ('data_cutoff', 'origin_at', 'forecast_start_at', 'forecast_end_at', 'opens_at', 'closes_at')
MONTHS_ES = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic']
MONTHS_EN = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def _ts(value):
    return pd.Timestamp(value).tz_convert('UTC')


# ---- fechas ----
DATES = {
    'fecha_espacio': lambda t: t.strftime('%Y-%m-%d %H:%M:%S'),
    'fecha_sin_zona': lambda t: t.strftime('%Y-%m-%dT%H:%M:%S'),
    'fecha_offset_bogota': lambda t: t.tz_convert('America/Bogota').isoformat(),
    'fecha_dmy': lambda t: t.strftime('%d/%m/%Y %H:%M'),
    'fecha_ymd_barra': lambda t: t.strftime('%Y/%m/%d %H:%M'),
    'fecha_milisegundos': lambda t: t.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
    'fecha_epoch_s': lambda t: int(t.timestamp()),
    'fecha_epoch_ms': lambda t: int(t.timestamp() * 1000),
    'fecha_epoch_texto': lambda t: str(int(t.timestamp())),
    'fecha_compacta': lambda t: t.strftime('%Y%m%dT%H%M%SZ'),
    'fecha_mes_es': lambda t: f'{t.day:02d} {MONTHS_ES[t.month - 1]} {t.year} {t:%H:%M}',
    'fecha_mes_en': lambda t: f'{MONTHS_EN[t.month - 1]} {t.day:02d}, {t.year} {t:%H:%M}',
    'fecha_rfc2822': lambda t: format_datetime(t.to_pydatetime(), usegmt=True),
    'fecha_utc_texto': lambda t: t.strftime('%Y-%m-%d %H:%M:%S UTC'),
    'fecha_am_pm': lambda t: t.strftime('%Y-%m-%d %I:%M %p'),
    'fecha_minusculas': lambda t: t.strftime('%Y-%m-%dt%H:%M:%Sz'),
    'fecha_offset_compacto': lambda t: t.tz_convert('America/Bogota').strftime('%Y-%m-%dT%H:%M:%S%z'),
}

# ---- números enteros ----
NUMBERS = {
    'numero_texto': lambda n: str(n),
    'numero_float': lambda n: float(n),
    'numero_texto_decimal': lambda n: f'{n}.0',
    'numero_coma_decimal': lambda n: f'{n},0',
    'numero_espacios': lambda n: f' {n} ',
    'numero_miles_coma': lambda n: f'{n:,}',
    'numero_miles_punto': lambda n: f'{n:,}'.replace(',', '.'),
    'numero_cientifico': lambda n: f'{n:e}',
}
# Horizontes como duración.
DURATIONS = {
    'horizonte_min': lambda n: f'{n} min',
    'horizonte_m': lambda n: f'{n}m',
    'horizonte_iso': lambda n: f'PT{n}M',
}

# ---- estaciones ----
STATIONS = {
    'estacion_entero': lambda s: int(s),
    'estacion_sin_ceros': lambda s: str(int(s)),
    'estacion_espacios': lambda s: f' {s} ',
    'estacion_float_texto': lambda s: f'{int(s)}.0',
    'estacion_prefijo': lambda s: f'ST-{s}',
}

# ---- booleanos (verdadero) ----
BOOLS = {
    'bool_yes': 'yes', 'bool_si': 'sí', 'bool_uno': 1, 'bool_texto_uno': '1', 'bool_true_texto': 'true', 'bool_Y': 'Y',
}

# ---- nombres de campos ----
KEYS = {
    'campos_camel': lambda k: ''.join(p if i == 0 else p.capitalize() for i, p in enumerate(k.split('_'))),
    'campos_mayusculas': str.upper,
}


def _map_dates_cycle(c, fn):
    for k in CYCLE_DATES:
        if c.get(k) is not None:
            c[k] = fn(_ts(c[k]))
    for t in c['targets']:
        t['target_at'] = fn(_ts(t['target_at']))
    return c


def _rename(obj, fn):
    if isinstance(obj, dict):
        return {fn(k): _rename(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rename(v, fn) for v in obj]
    return obj


def cycle_mutations(raw):
    """{nombre: ciclo mutado}."""
    out = {}
    for name, fn in DATES.items():
        out[name] = _map_dates_cycle(copy.deepcopy(raw), fn)
    for name, fn in NUMBERS.items():
        c = copy.deepcopy(raw)
        for k in ('expected_predictions', 'station_count'):
            if k in c:
                c[k] = fn(c[k])
        c['horizons_minutes'] = [fn(h) for h in c.get('horizons_minutes', [])]
        for t in c['targets']:
            t['horizon_minutes'] = fn(t['horizon_minutes'])
        out[name] = c
    for name, fn in STATIONS.items():
        c = copy.deepcopy(raw)
        for t in c['targets']:
            t['station_id'] = fn(t['station_id'])
        out[name] = c
    for name, fn in KEYS.items():
        out[name] = _rename(copy.deepcopy(raw), fn)
    for name, fn in DURATIONS.items():
        c = copy.deepcopy(raw)
        c['horizons_minutes'] = [fn(h) for h in c.get('horizons_minutes', [])]
        for t in c['targets']:
            t['horizon_minutes'] = fn(t['horizon_minutes'])
        out[name] = c
    c = copy.deepcopy(raw); c['items'] = c.pop('targets'); out['targets_como_items'] = c
    c = copy.deepcopy(raw); out['ciclo_envuelto'] = {'data': c}
    c = copy.deepcopy(raw); c['state'] = 'OPEN'; out['estado_mayusculas'] = c
    c = copy.deepcopy(raw); c['state'] = True; out['estado_booleano'] = c
    c = copy.deepcopy(raw); c['state'] = 'yes'; out['estado_yes'] = c
    c = copy.deepcopy(raw); c['extra_campo'] = {'nuevo': [1, 2]}; out['campo_extra'] = c
    # Peor caso: varias a la vez.
    c = _map_dates_cycle(copy.deepcopy(raw), DATES['fecha_dmy'])
    c['expected_predictions'] = str(c['expected_predictions'])
    c['horizons_minutes'] = [str(h) for h in c.get('horizons_minutes', [])]
    for t in c['targets']:
        t['horizon_minutes'] = str(t['horizon_minutes']); t['station_id'] = int(t['station_id'])
    out['combinada'] = c
    return out


def stream_mutations(page):
    out = {}
    for name, fn in DATES.items():
        p = copy.deepcopy(page)
        for r in p['data']:
            r['observed_at'] = fn(_ts(r['observed_at'])); r['released_at'] = fn(_ts(r['released_at']))
        out[name] = p
    for name, fn in NUMBERS.items():
        p = copy.deepcopy(page)
        for r in p['data']:
            r['demand'] = fn(r['demand'])
        out[name] = p
    for name, fn in STATIONS.items():
        p = copy.deepcopy(page)
        for r in p['data']:
            r['station_id'] = fn(r['station_id'])
        out[name] = p
    for name, fn in KEYS.items():
        p = copy.deepcopy(page); p['data'] = _rename(p['data'], fn); out[name] = p
    names = {'station_id': 'id_est', 'observed_at': 'ts', 'demand': 'valor', 'released_at': 'pub'}
    p = copy.deepcopy(page); p['data'] = [{names[k]: v for k, v in r.items()} for r in p['data']]
    out['campos_desconocidos'] = p
    p = copy.deepcopy(page); p['rows'] = p.pop('data'); out['pagina_rows'] = p
    p = copy.deepcopy(page); out['pagina_anidada'] = {'result': {'data': p.pop('data'), **p}}
    p = copy.deepcopy(page); p['items'] = p.pop('data'); out['pagina_items'] = p
    p = copy.deepcopy(page); p['nextCursor'] = p.pop('next_cursor'); out['pagina_cursor_camel'] = p
    return out


def receipt_mutations(body):
    out = {}
    for name, value in BOOLS.items():
        b = copy.deepcopy(body); b['is_official'] = value; out[name] = b
    for name, value in (('estado_mayusculas', 'ACCEPTED'), ('estado_espacios', ' accepted '),
                        ('estado_booleano', True), ('estado_yes', 'yes'), ('estado_es', 'aceptado'), ('estado_ok', 'ok')):
        b = copy.deepcopy(body); b['status'] = value; out[f'recibo_{name}'] = b
    b = _rename(copy.deepcopy(body), KEYS['campos_camel']); out['recibo_campos_camel'] = b
    return out


REAL_RECEIPT = {'status': 'accepted', 'attempt': 1, 'closes_at': '2026-10-03T02:14:57.894738+00:00',
                'is_official': True, 'received_at': '2026-10-03T01:50:10.395838+00:00',
                'payload_hash': 'sha256:' + 'b' * 64, 'submission_id': 'sub_b6762a9003f44418b82c3a5eff739047'}


def now_utc():
    return datetime.now(timezone.utc)
