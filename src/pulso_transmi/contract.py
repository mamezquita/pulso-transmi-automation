"""Capa de contrato: traduce datos de la API a tipos internos fijos.

El profesor puede cambiar formatos (orden de fecha, tipos, nombres de campos).
Aquí se aceptan variantes inequívocas y se rechaza lo ambiguo, nunca se adivina
en silencio. El formato ISO actual pasa por la ruta rápida sin cambios.
"""
from datetime import datetime, timezone
import math
import re

import pandas as pd

GRID_NS = 900_000_000_000  # 15 minutos
ISO = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:\d{2})')
GENERIC = re.compile(
    r'(\d{1,4})[-/.](\d{1,2})[-/.](\d{1,4})'
    r'(?:[T ](\d{1,2}):(\d{2})(?::(\d{2})(?:\.(\d{1,9}))?)?)?'
    r'\s*(Z|UTC|[+-]\d{2}:?\d{2})?')
# Orden preferido primero. Y-m-d es el estándar; Y-d-m casi nunca se usa.
ORDERS = {'ymd': (0, 1, 2), 'ydm': (0, 2, 1), 'dmy': (2, 1, 0), 'mdy': (2, 0, 1)}
ALIASES = {
    'station_id': ('station_id', 'stationId', 'estacion_id', 'station', 'estacion'),
    'observed_at': ('observed_at', 'observedAt', 'observado_en', 'timestamp', 'fecha'),
    'demand': ('demand', 'demanda', 'value', 'count', 'passengers'),
    'released_at': ('released_at', 'releasedAt', 'publicado_en', 'published_at'),
}


CYCLE_FIELDS = ('cycle_id', 'state', 'origin_at', 'data_cutoff', 'opens_at', 'closes_at', 'forecast_start_at',
                'forecast_end_at', 'station_count', 'horizons_minutes', 'expected_predictions', 'targets')
TARGET_FIELDS = ('station_id', 'target_at', 'horizon_minutes')
RECEIPT_FIELDS = ('status', 'submission_id', 'received_at', 'closes_at', 'attempt', 'is_official', 'payload_hash')
_MONTH_NAMES = [
    ('ene', 'enero', 'jan', 'january'), ('feb', 'febrero', 'february'), ('mar', 'marzo', 'march'),
    ('abr', 'abril', 'apr', 'april'), ('may', 'mayo'), ('jun', 'junio', 'june'), ('jul', 'julio', 'july'),
    ('ago', 'agosto', 'aug', 'august'), ('sep', 'sept', 'septiembre', 'setiembre', 'september'),
    ('oct', 'octubre', 'october'), ('nov', 'noviembre', 'november'), ('dic', 'diciembre', 'dec', 'december')]
MONTHS = {m: i + 1 for i, names in enumerate(_MONTH_NAMES) for m in names}
TRUE_WORDS = {'true', 't', 'yes', 'y', 'si', 'sí', 's', '1', 'open', 'opened', 'abierto', 'abierta', 'activo', 'on'}
FALSE_WORDS = {'false', 'f', 'no', 'n', '0', 'closed', 'close', 'cerrado', 'cerrada', 'inactivo', 'off'}
ACCEPTED_WORDS = {'accepted', 'aceptado', 'aceptada', 'ok', 'success', 'successful', 'exito', 'éxito', 'received',
                  'recibido'}


def norm_key(key):
    """`observedAt`, `OBSERVED_AT` y `observed-at` son el mismo campo."""
    return re.sub(r'[^a-z0-9]', '', str(key).lower())


def canonical_keys(obj, names):
    """Renombra a su forma canónica las claves que solo difieren en formato."""
    if not isinstance(obj, dict):
        return obj
    lookup = {norm_key(n): n for n in names}
    out = {}
    for k, v in obj.items():
        out[lookup.get(norm_key(k), k)] = v
    return out


def boolean(value):
    """True/False desde 1/0, yes/no, sí/no, true/false, open/closed…; None si no se reconoce."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    return True if text in TRUE_WORDS else False if text in FALSE_WORDS else None


class ContractError(ValueError):
    """Dato que no se puede interpretar de forma inequívoca."""


def canonical(ts):
    """Formato de salida único; idéntico al que la API usa hoy para tiempo simulado."""
    ts = ts.tz_convert('UTC')
    if ts.microsecond or ts.nanosecond:
        return ts.isoformat().replace('+00:00', 'Z')
    return ts.strftime('%Y-%m-%dT%H:%M:%SZ')


def pattern(value):
    """Huella del formato para detectar cambios: dígitos -> 9."""
    if isinstance(value, bool) or value is None:
        return type(value).__name__
    if isinstance(value, (int, float)):
        return type(value).__name__
    if re.fullmatch(r'\s*-?\d+(\.\d+)?\s*', str(value)):
        return 'str-num'
    return re.sub(r'\d', '9', str(value))


def _epoch(value):
    if isinstance(value, bool) or not math.isfinite(value):
        raise ContractError('Fecha numérica inválida')
    unit = 's' if abs(value) < 1e11 else 'ms' if abs(value) < 1e14 else 'us' if abs(value) < 1e17 else 'ns'
    return pd.Timestamp(value, unit=unit, tz='UTC')


def _candidates(text):
    """Lecturas válidas de una fecha no ISO, en orden de preferencia."""
    m = GENERIC.fullmatch(text.strip())
    if not m:
        raise ContractError('Formato de fecha no reconocido')
    parts = m.groups()[:3]
    hour, minute, second, frac, tz = m.groups()[3:]
    out = {}
    for name, (iy, im, id_) in ORDERS.items():
        if len(parts[iy]) != 4 or len(parts[im]) > 2 or len(parts[id_]) > 2:
            continue
        try:
            naive = datetime(int(parts[iy]), int(parts[im]), int(parts[id_]), int(hour or 0),
                             int(minute or 0), int(second or 0), int((frac or '0').ljust(6, '0')[:6]))
        except ValueError:
            continue
        ts = pd.Timestamp(naive)
        if tz in (None, 'Z', 'UTC'):
            ts = ts.tz_localize('UTC')
        else:
            ts = ts.tz_localize(datetime.strptime(tz.replace(':', ''), '%z').tzinfo).tz_convert('UTC')
        out.setdefault(name, ts)
    if not out:
        raise ContractError('Fecha inválida en todos los órdenes')
    return out


def parse_instants(values, near=None):
    """Interpreta una columna de fechas; un mismo orden por cada formato.

    Los valores se agrupan por huella de formato. Dentro de un grupo, si algún
    valor descarta un orden (p. ej. día 13), se descarta para todo el grupo. Si
    quedan varios órdenes, `near` desempata; sin él solo se acepta año-mes-día.
    Lo demás es ambiguo y se rechaza. Fechas sin zona se asumen UTC, como hoy.
    """
    values = list(values)
    if all(isinstance(v, str) and ISO.fullmatch(v.strip()) for v in values):
        return [pd.Timestamp(v.strip()).tz_convert('UTC') for v in values]
    groups = {}
    for i, v in enumerate(values):
        groups.setdefault(pattern(v), []).append(i)
    out = [None] * len(values)
    for idx in groups.values():
        for i, ts in zip(idx, _parse_group([values[i] for i in idx], near)):
            out[i] = ts
    return out


COMPACT = re.compile(r'(\d{4})(\d{2})(\d{2})[T ]?(\d{2})(\d{2})(\d{2})?(\.\d+)?\s*(Z|UTC|[+-]\d{2}:?\d{2})?')
WORD_DATE = re.compile(
    r'(?:(\d{1,2})[\s./-]+([a-záéíóú]+)\.?[\s./-]+(\d{4})|([a-záéíóú]+)\.?\s+(\d{1,2}),?\s+(\d{4}))'
    r'(?:[\sT,]+(\d{1,2}):(\d{2})(?::(\d{2}))?)?\s*(Z|UTC|[+-]\d{2}:?\d{2})?')


def _rewrite(text):
    """Formatos inequívocos que se reescriben como año-mes-día antes de interpretarlos."""
    t = text.strip()
    m = COMPACT.fullmatch(t)
    if m:
        y, mo, d, h, mi, sec, frac, tz = m.groups()
        return f"{y}-{mo}-{d}T{h}:{mi}:{sec or '00'}{frac or ''}{(tz or '').replace('UTC', 'Z')}"
    m = WORD_DATE.fullmatch(t.lower())
    if m:
        d1, mon1, y1, mon2, d2, y2, h, mi, sec, tz = m.groups()
        month = MONTHS.get(mon1 or mon2)
        if month is None:
            raise ContractError('Mes no reconocido')
        tz = (tz or '').upper().replace('UTC', 'Z')
        return f"{y1 or y2}-{month:02d}-{int(d1 or d2):02d} {int(h or 0):02d}:{mi or '00'}:{sec or '00'}{tz}"
    return t


def _parse_group(values, near):
    if all(isinstance(v, str) and ISO.fullmatch(v.strip()) for v in values):
        return [pd.Timestamp(v.strip()).tz_convert('UTC') for v in values]
    if all(isinstance(v, str) and re.fullmatch(r'\s*\d{9,19}(\.\d+)?\s*', v) for v in values):
        return [_epoch(float(v)) for v in values]  # epoch como texto
    if all(isinstance(v, str) for v in values):
        values = [_rewrite(v) for v in values]
        if all(ISO.fullmatch(v) for v in values):
            return [pd.Timestamp(v).tz_convert('UTC') for v in values]
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return [_epoch(v) for v in values]
    if not all(isinstance(v, str) for v in values):
        raise ContractError('Fecha con tipo inválido')
    per_value = [_candidates(v) for v in values]
    orders = [o for o in ORDERS if all(o in c for c in per_value)]
    if not orders:
        raise ContractError('Ningún orden de fecha es válido para todo el formato')
    distinct = {}
    for o in orders:
        parsed = [c[o] for c in per_value]
        if parsed not in distinct.values():
            distinct[o] = parsed
    if len(distinct) == 1:
        return next(iter(distinct.values()))
    if near is not None:
        near = pd.Timestamp(near).tz_convert('UTC')
        dist = {o: abs(pd.Series(p) - near).median() for o, p in distinct.items()}
        ranked = sorted(dist, key=dist.get)
        if dist[ranked[0]] < dist[ranked[1]]:
            return distinct[ranked[0]]
    # Sin referencia útil: año primero se lee como año-mes-día (estándar ISO).
    if 'ymd' in distinct and all(len(GENERIC.fullmatch(v.strip()).group(1)) == 4 for v in values):
        return distinct['ymd']
    raise ContractError('Fecha ambigua (día/mes); se requiere una referencia')


def station(value):
    if isinstance(value, bool):
        raise ContractError('Estación inválida')
    if isinstance(value, int) and 0 <= value < 100_000:
        return f'{value:05d}'
    if isinstance(value, float) and value.is_integer() and 0 <= value < 100_000:
        return f'{int(value):05d}'
    if isinstance(value, str) and re.fullmatch(r'\d{1,5}', value.strip()):
        return value.strip().zfill(5)
    raise ContractError('Estación inválida')


def count(value):
    """Entero no negativo exacto; acepta 12, 12.0, "12", "12.0" y "12,0" pero no 12.5."""
    if isinstance(value, bool):
        raise ContractError('Demanda inválida')
    if isinstance(value, int):
        number = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        number = int(value)
    elif isinstance(value, str) and re.fullmatch(r'\s*\d+([.,]0+)?\s*', value):
        number = int(value.strip().replace(',', '.').split('.')[0])
    else:
        raise ContractError('Demanda inválida')
    if number < 0:
        raise ContractError('Demanda negativa')
    return number


def field_key(row, name):
    """Clave de `row` que corresponde al campo `name` (alias y variantes de formato)."""
    wanted = {norm_key(a) for a in ALIASES[name]}
    for key in row:
        if norm_key(key) in wanted:
            return key
    raise ContractError(f'Falta el campo {name}')


def field(row, name):
    return row[field_key(row, name)]


def normalize_observations(rows, near=None):
    """Devuelve (filas_canónicas, cuarentena, huella). Nunca lanza por una fila.

    La cuarentena guarda la fila original y el motivo para reprocesarla.
    """
    good, quarantine, fingerprint = [], [], {}
    staged = []
    for raw in rows:
        try:
            if not isinstance(raw, dict):
                raise ContractError('Fila no es un objeto')
            staged.append((raw, station(field(raw, 'station_id')), count(field(raw, 'demand')),
                           field(raw, 'observed_at'), field(raw, 'released_at')))
            for name in ALIASES:
                key = field_key(raw, name)
                fingerprint.setdefault(name, set()).add(f'{key}:{pattern(raw[key])}')
        except ContractError as exc:
            quarantine.append({'row': raw, 'reason': str(exc)})
    if staged:
        try:
            observed = parse_instants([s[3] for s in staged], near=near)
            released = parse_instants([s[4] for s in staged], near=datetime.now(timezone.utc))
        except ContractError as exc:
            quarantine += [{'row': s[0], 'reason': str(exc)} for s in staged]
            staged = []
        for i, (raw, sid, demand, _, _) in enumerate(staged):
            if observed[i].value % GRID_NS:
                quarantine.append({'row': raw, 'reason': 'Timestamp desalineado'})
                continue
            good.append({'station_id': sid, 'observed_at': canonical(observed[i]),
                         'demand': demand, 'released_at': canonical(released[i])})
    return good, quarantine, {k: sorted(v) for k, v in fingerprint.items()}


def _cycle_reference(cycle_id):
    m = re.search(r'(\d{8}T\d{6}Z)', str(cycle_id))
    return pd.Timestamp(datetime.strptime(m.group(1), '%Y%m%dT%H%M%SZ'), tz='UTC') if m else None


def normalize_cycle_contract(raw):
    """Contrato del ciclo en formato canónico, verificado por consistencia interna."""
    c = canonical_keys(dict(raw), CYCLE_FIELDS)
    if not isinstance(c.get('cycle_id'), str) or not isinstance(c.get('targets'), list) or not c['targets']:
        raise ContractError('Contrato sin cycle_id o targets')
    state = boolean(c.get('state'))
    c['state'] = 'open' if state is True else 'closed' if state is False else str(c.get('state', '')).strip().lower()
    ref = _cycle_reference(c['cycle_id'])
    sim_keys = [k for k in ('data_cutoff', 'origin_at', 'forecast_start_at', 'forecast_end_at') if c.get(k) is not None]
    targets = [canonical_keys(dict(t), TARGET_FIELDS) for t in c['targets']]
    try:
        parsed = parse_instants([c[k] for k in sim_keys] + [t.get('target_at') for t in targets], near=ref)
    except ContractError as exc:
        parsed = _rebuild_instants(c, sim_keys, targets, ref, exc)
    for k, ts in zip(sim_keys, parsed):
        c[k] = canonical(ts)
    cutoff = parsed[sim_keys.index('data_cutoff')] if 'data_cutoff' in sim_keys else None
    if cutoff is None:
        raise ContractError('Contrato sin data_cutoff')
    horizons = sorted({int(count(h)) for h in c.get('horizons_minutes') or []})
    for t, ts in zip(targets, parsed[len(sim_keys):]):
        t['station_id'] = station(t.get('station_id'))
        t['target_at'] = canonical(ts)
        lead = (ts - cutoff).total_seconds() / 60
        if 'horizon_minutes' in t:
            t['horizon_minutes'] = count(t['horizon_minutes'])
            if lead != t['horizon_minutes']:
                raise ContractError('target_at no coincide con horizon_minutes')
        if horizons and lead not in horizons:
            raise ContractError('target_at fuera de los horizontes del contrato')
    c['targets'] = targets
    if horizons:
        c['horizons_minutes'] = horizons
    for k in ('expected_predictions', 'station_count'):
        if k in c:
            c[k] = count(c[k])
    for k in ('opens_at', 'closes_at'):
        if c.get(k) is None:
            continue
        try:
            c[k] = canonical(parse_instants([c[k]], near=datetime.now(timezone.utc))[0])
        except ContractError:
            # Hora real ilegible: no se inventa. El envío confía en la respuesta del servidor.
            c.setdefault('formato_no_leido', []).append(k)
            c[k] = None
    return c


def _rebuild_instants(c, sim_keys, targets, ref, exc):
    """Si las fechas del contrato no se pueden leer, se reconstruyen con lo que es fijo:
    el corte viene en el `cycle_id` y cada objetivo es corte + su horizonte. Solo si todo
    objetivo trae su horizonte legible; si no, el error original se mantiene."""
    if ref is None:
        raise exc
    try:
        horizons = [count(t['horizon_minutes']) for t in targets]
    except (KeyError, ContractError):
        raise exc
    by_key = {}
    for k in sim_keys:
        try:
            by_key[k] = parse_instants([c[k]], near=ref)[0]
        except ContractError:
            by_key[k] = None
    if by_key.get('data_cutoff') not in (None, ref):
        raise ContractError('data_cutoff legible no coincide con el cycle_id')
    by_key['data_cutoff'] = ref
    known = {'origin_at': ref, 'forecast_start_at': ref + pd.Timedelta(minutes=min(horizons)),
             'forecast_end_at': ref + pd.Timedelta(minutes=max(horizons))}
    c['reconstruido'] = f'fechas reconstruidas desde cycle_id y horizontes ({exc})'
    return [by_key[k] if by_key[k] is not None else known[k] for k in sim_keys] + \
           [ref + pd.Timedelta(minutes=h) for h in horizons]


def normalize_receipt(body, http_status):
    """Recibo del POST en forma canónica: `status` = 'accepted' si el servidor lo aceptó
    (2xx y estado afirmativo en cualquier formato) y `submission_id` con su nombre canónico.
    El original se conserva en `raw` para diagnóstico."""
    if not isinstance(body, dict):
        return body
    b = canonical_keys(dict(body), RECEIPT_FIELDS)
    status = b.get('status')
    text = str(status).strip().lower()
    if 200 <= int(http_status) < 300 and (boolean(status) is True or text in ACCEPTED_WORDS):
        b['status'] = 'accepted'
    if b.get('submission_id') is not None:
        b['submission_id'] = str(b['submission_id']).strip()
    if b != body:
        b['raw'] = body
    return b
