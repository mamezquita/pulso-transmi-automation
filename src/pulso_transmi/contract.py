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


RFC2822 = re.compile(r'[A-Za-z]{3},\s*\d{1,2}\s+[A-Za-z]{3}\s+\d{4}\s+\d{1,2}:\d{2}(:\d{2})?\s*(GMT|UTC|Z|[+-]\d{4})?')
AM_PM = re.compile(r'(.*?)(\d{1,2})(:\d{2}(?::\d{2})?)\s*([ap])\.?\s*m\.?\s*(Z|UTC|GMT|[+-]\d{2}:?\d{2})?', re.I)


def _rewrite(text):
    """Formatos inequívocos que se reescriben como año-mes-día antes de interpretarlos."""
    t = text.strip()
    if RFC2822.fullmatch(t):  # Sun, 20 Sep 2026 09:00:00 GMT
        from email.utils import parsedate_to_datetime
        try:
            dt = parsedate_to_datetime(t)
        except (TypeError, ValueError):
            raise ContractError('Fecha RFC 2822 inválida')
        return pd.Timestamp(dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).isoformat()
    m = AM_PM.fullmatch(t)
    if m:  # 09:00 PM -> 21:00
        head, hour, rest, half, tz = m.groups()
        hour = int(hour) % 12 + (12 if half.lower() == 'p' else 0)
        t = f'{head}{hour:02d}{rest}{tz or ""}'
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}t\d{2}:\d{2}.*', t):  # t/z en minúscula
        t = t.upper()
    if t.endswith(' GMT'):
        t = t[:-4] + 'Z'
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
    if isinstance(value, str):
        # "02300", " 2300 ", "2300.0", "ST-02300", "estacion 02300"
        m = re.fullmatch(r'\s*(?:[A-Za-zÁÉÍÓÚáéíóúñ]+[\s_#:-]*)?(\d{1,5})(?:\.0+)?\s*', value)
        if m:
            return m.group(1).zfill(5)
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
    elif isinstance(value, str) and re.fullmatch(r'\s*\d{1,3}(,\d{3})+\s*', value):
        number = int(value.strip().replace(',', ''))  # miles con coma: 1,234
    elif isinstance(value, str) and re.fullmatch(r'\s*\d{1,3}(\.\d{3})+\s*', value):
        number = int(value.strip().replace('.', ''))  # miles con punto: 1.234 (1.000 ya se leyó como 1)
    elif isinstance(value, str) and re.fullmatch(r'\s*\d+(\.\d+)?[eE][+-]?\d+\s*', value):
        number = float(value)  # notación científica: 1.41e+02
        if not number.is_integer():
            raise ContractError('Demanda inválida')
        number = int(number)
    else:
        raise ContractError('Demanda inválida')
    if number < 0:
        raise ContractError('Demanda negativa')
    return number


def minutes(value):
    """Duración en minutos: 15, "15", "15 min", "15m", "PT15M", "00:15"."""
    if isinstance(value, str):
        t = value.strip().lower()
        m = (re.fullmatch(r'(\d+)\s*(?:m|min|mins|minutos?|minutes?)', t)
             or re.fullmatch(r'pt(\d+)m', t))
        if m:
            return int(m.group(1))
        m = re.fullmatch(r'(\d{1,2}):(\d{2})', t)
        if m:
            return int(m.group(1)) * 60 + int(m.group(2))
    return count(value)


def field_key(row, name):
    """Clave de `row` que corresponde al campo `name` (alias y variantes de formato)."""
    wanted = {norm_key(a) for a in ALIASES[name]}
    for key in row:
        if norm_key(key) in wanted:
            return key
    raise ContractError(f'Falta el campo {name}')


def field(row, name):
    return row[field_key(row, name)]


MISSING_QUALITY = {'missing', 'faltante', 'null', 'na', 'n/a', 'unavailable'}
MISSING_REASON = 'Valor faltante reportado por la API'
PAGE_ROWS = ('data', 'items', 'results', 'rows', 'records', 'observations', 'observaciones')


def page_parts(page, depth=0):
    """(filas, next_cursor) de una página aunque cambie su envoltura: `data`, `items`,
    `rows`…, una única lista de objetos, o todo dentro de un objeto (un nivel)."""
    if isinstance(page, list):
        return page, None
    if not isinstance(page, dict):
        raise ContractError('Página inválida')
    p = canonical_keys(page, PAGE_ROWS + ('next_cursor',))
    for k in PAGE_ROWS:
        if isinstance(p.get(k), list):
            return p[k], p.get('next_cursor')
    lists = [v for v in p.values() if isinstance(v, list) and all(isinstance(x, dict) for x in v)]
    if len(lists) == 1:
        return lists[0], p.get('next_cursor')
    inner = [v for v in p.values() if isinstance(v, dict)]
    if depth == 0 and len(inner) == 1:
        rows, cursor = page_parts(inner[0], 1)
        return rows, cursor if cursor is not None else p.get('next_cursor')
    raise ContractError('Página sin filas reconocibles')


def _infer_columns(rows):
    """Si los nombres de campos no coinciden con ningún alias, reconoce cada columna por su
    contenido: dos columnas de fechas (la observación va en la grilla de 15 min y es la más
    antigua), códigos de estación y conteos. Solo devuelve lo que queda sin ambigüedad."""
    sample = [r for r in rows if isinstance(r, dict)][:200]
    if not sample:
        return {}
    keys = [k for k in sample[0] if all(k in r for r in sample)]
    dates, stations, counts = {}, [], []
    for k in keys:
        values = [r[k] for r in sample]
        try:
            parsed = parse_instants(values)
            if all(isinstance(v, str) or (isinstance(v, (int, float)) and abs(v) >= 1e9) for v in values):
                dates[k] = parsed
                continue
        except (ContractError, ValueError, TypeError, OverflowError):
            pass
        try:
            [station(v) for v in values]; stations.append(k)
        except ContractError:
            pass
        try:
            [count(v) for v in values]; counts.append(k)
        except ContractError:
            pass
    out = {}
    if len(dates) in (1, 2):
        aligned = [k for k, p in dates.items() if all(ts.value % GRID_NS == 0 for ts in p)]
        if len(dates) == 1:
            out['observed_at'] = out['released_at'] = next(iter(dates))
        else:
            a, b = dates
            first = a if pd.Series(dates[a]).median() <= pd.Series(dates[b]).median() else b
            out['observed_at'] = first if first in aligned or len(aligned) != 1 else aligned[0]
            out['released_at'] = b if out['observed_at'] == a else a
    numeric = [k for k in keys if k not in dates]
    if len(numeric) == 2 and set(numeric) <= set(stations) | set(counts):
        # Estación: la columna con ceros a la izquierda o, si no, con menos valores distintos.
        zero = [k for k in numeric if any(isinstance(r[k], str) and r[k].strip().startswith('0') for r in sample)]
        distinct = {k: len({str(r[k]) for r in sample}) for k in numeric}
        st = zero[0] if len(zero) == 1 else min(numeric, key=distinct.get) if len(set(distinct.values())) == 2 else None
        dm = next((k for k in numeric if k != st), None)
        if st in stations and dm in counts:
            out['station_id'], out['demand'] = st, dm
    return out


def flatten(row):
    """Sube un nivel los objetos anidados: {"measurement": {"value": 5}} agrega
    `measurement.value` y, si no choca con un campo existente, también `value`."""
    if not isinstance(row, dict) or not any(isinstance(v, dict) for v in row.values()):
        return row
    out = {k: v for k, v in row.items() if not isinstance(v, dict)}
    for k, v in row.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[f'{k}.{kk}'] = vv
                out.setdefault(kk, vv)
            wrapped = next((v[w] for w in ('value', 'valor') if w in v), None)
            if wrapped is not None and not isinstance(wrapped, (dict, list)):
                out.setdefault(k, wrapped)  # {"horizon_minutes": {"value": 15}} -> 15
    return out


UNITS = {'passengers': 1, 'pasajeros': 1, 'pax': 1, 'personas': 1, 'people': 1, 'count': 1, 'unidades': 1,
         'hundreds': 100, 'cientos': 100, 'centenas': 100, 'thousands': 1000, 'miles': 1000, 'k': 1000}


def scaled_count(value, unit):
    """Demanda en pasajeros aunque venga en otra unidad conocida ("5.46" cientos -> 546).
    Una unidad desconocida no se adivina: va a cuarentena."""
    if unit is None:
        return count(value)
    factor = UNITS.get(re.sub(r'[^a-z]', '', str(unit).lower()))
    if factor is None:
        raise ContractError(f'Unidad desconocida: {unit}')
    if factor == 1:
        return count(value)
    try:
        number = float(str(value).strip().replace(',', '.')) * factor
    except ValueError:
        raise ContractError('Demanda inválida')
    if not math.isfinite(number) or number < 0 or abs(number - round(number)) > 1e-6 * max(1, number):
        raise ContractError('Demanda inválida')
    return int(round(number))


def normalize_observations(rows, near=None):
    """Devuelve (filas_canónicas, cuarentena, huella). Nunca lanza por una fila.

    La cuarentena guarda la fila original y el motivo para reprocesarla.
    """
    good, quarantine, fingerprint = [], [], {}
    staged = []
    inferred = None
    originals = rows
    rows = [flatten(r) for r in rows]  # schema 2 del stream: demanda en measurement.value

    def get_key(raw, name):
        nonlocal inferred
        try:
            return field_key(raw, name)
        except ContractError:
            if inferred is None:
                inferred = _infer_columns(rows)
            if inferred.get(name) in raw:
                return inferred[name]
            raise

    for original, raw in zip(originals, rows):
        try:
            if not isinstance(raw, dict):
                raise ContractError('Fila no es un objeto')
            if str(raw.get('quality', '')).strip().lower() in MISSING_QUALITY and raw.get('value') is None:
                raise ContractError(MISSING_REASON)  # la API reporta el dato como faltante
            keys = {name: get_key(raw, name) for name in ALIASES}
            staged.append((original, station(raw[keys['station_id']]), scaled_count(raw[keys['demand']], raw.get('unit')),
                           raw[keys['observed_at']], raw[keys['released_at']]))
            for name, key in keys.items():
                fingerprint.setdefault(name, set()).add(f'{key}:{pattern(raw[key])}')
        except ContractError as exc:
            quarantine.append({'row': original, 'reason': str(exc)})
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


TARGET_LISTS = ('targets', 'items', 'objetivos', 'predictions', 'required_predictions', 'pronosticos', 'data')


def _unwrap_cycle(raw, depth=0):
    """Contrato con claves canónicas aunque venga dentro de otro objeto o con `targets`
    bajo otro nombre (o como la única lista de objetos con estación)."""
    if not isinstance(raw, dict):
        raise ContractError('Contrato inválido')
    c = canonical_keys(dict(raw), CYCLE_FIELDS)
    if 'cycle_id' not in c and depth == 0:
        inner = [v for v in c.values() if isinstance(v, dict) and 'cycleid' in {norm_key(k) for k in v}]
        if len(inner) == 1:
            return _unwrap_cycle(inner[0], 1)
    c = canonical_keys(flatten(c), CYCLE_FIELDS)  # secciones anidadas: {"cycle": {"id": …}}
    if 'cycle_id' not in c and isinstance(c.get('id'), str):
        c['cycle_id'] = c['id']
    if 'state' not in c:
        for alt in ('status', 'estado', 'is_open', 'open', 'abierto'):
            if alt in c:
                c['state'] = c[alt]
                break
    if not isinstance(c.get('targets'), list):
        named = {norm_key(n) for n in TARGET_LISTS}
        found = [k for k, v in c.items() if norm_key(k) in named and isinstance(v, list)]
        if not found:
            found = [k for k, v in c.items() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)
                     and all('station' in ''.join(norm_key(kk) for kk in x) or 'estacion' in ''.join(norm_key(kk) for kk in x)
                             for x in v)]
        if len(found) == 1:
            c['targets'] = c.pop(found[0])
    return c


def _cycle_reference(cycle_id):
    m = re.search(r'(\d{8}T\d{6}Z)', str(cycle_id))
    return pd.Timestamp(datetime.strptime(m.group(1), '%Y%m%dT%H%M%SZ'), tz='UTC') if m else None


def normalize_cycle_contract(raw):
    """Contrato del ciclo en formato canónico, verificado por consistencia interna."""
    c = _unwrap_cycle(raw)
    if not isinstance(c.get('cycle_id'), str) or not isinstance(c.get('targets'), list) or not c['targets']:
        raise ContractError('Contrato sin cycle_id o targets')
    state = boolean(c.get('state'))
    c['state'] = 'open' if state is True else 'closed' if state is False else str(c.get('state', '')).strip().lower()
    ref = _cycle_reference(c['cycle_id'])
    sim_keys = [k for k in ('data_cutoff', 'origin_at', 'forecast_start_at', 'forecast_end_at') if c.get(k) is not None]
    targets = [canonical_keys(flatten(dict(t)), TARGET_FIELDS) for t in c['targets']]
    try:
        parsed = parse_instants([c[k] for k in sim_keys] + [t.get('target_at') for t in targets], near=ref)
    except ContractError as exc:
        parsed = _rebuild_instants(c, sim_keys, targets, ref, exc)
    for k, ts in zip(sim_keys, parsed):
        c[k] = canonical(ts)
    cutoff = parsed[sim_keys.index('data_cutoff')] if 'data_cutoff' in sim_keys else None
    if cutoff is None:
        raise ContractError('Contrato sin data_cutoff')
    horizons = sorted({minutes(h) for h in c.get('horizons_minutes') or []})
    for t, ts in zip(targets, parsed[len(sim_keys):]):
        t['station_id'] = station(t.get('station_id'))
        t['target_at'] = canonical(ts)
        lead = (ts - cutoff).total_seconds() / 60
        if 'horizon_minutes' in t:
            t['horizon_minutes'] = minutes(t['horizon_minutes'])
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
    if 'expected_predictions' not in c:
        c['expected_predictions'] = len(targets)  # cada objetivo es una predicción esperada
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
        horizons = [minutes(t['horizon_minutes']) for t in targets]
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
