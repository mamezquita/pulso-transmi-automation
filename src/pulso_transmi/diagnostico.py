"""Diagnóstico de cambios de formato de la API.

Cada intento guarda, por fuente (ciclo, stream, recibo), la "forma" de la respuesta
cruda: por cada ruta de campo, su tipo o patrón (dígitos -> 9, letras -> a). Supabase
(`registrar_formato`) solo agrega una fila cuando la forma cambia, con una muestra cruda
y la forma anterior, así se ve exactamente qué cambió y cuándo. Nunca bloquea el envío.
"""
import re

_captured = {}
LIST_SAMPLE = 1000  # todas las filas de la página: el 3-oct el cambio venía después de las primeras
SAMPLE_ROWS = 3


def capture(source, raw):
    """Guarda la respuesta cruda de una fuente para registrarla al final del intento."""
    _captured.setdefault(source, raw)


def _scalar(value):
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'bool'
    if isinstance(value, int):
        return 'int'
    if isinstance(value, float):
        return 'float'
    text = str(value)
    if len(text) > 60:
        return 'texto-largo'
    if len(text) <= 20 and not re.search(r'\d', text):
        return 'txt=' + text  # valores tipo categoría (status, state): el valor es el formato
    text = re.sub(r'[0-9a-f]{16,}', 'H', text)  # hashes e identificadores aleatorios
    return 'str:' + re.sub(r'[a-zA-ZáéíóúñÁÉÍÓÚÑ]', 'a', re.sub(r'\d', '9', text))


def shape(obj, prefix=''):
    """{ruta: patrón}. En listas se unen los patrones de los primeros elementos."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            for path, pat in shape(v, f'{prefix}.{k}' if prefix else str(k)).items():
                out[path] = pat
    elif isinstance(obj, list):
        merged = {}
        for item in obj[:LIST_SAMPLE]:
            for path, pat in shape(item, prefix + '[]').items():
                merged.setdefault(path, set()).add(pat)
        out = {p: '|'.join(sorted(v)) for p, v in merged.items()}
    elif obj is not None:  # nulos (p. ej. next_cursor al final) no son un cambio de formato
        out[prefix or '.'] = _scalar(obj)
    return out


def _sample(raw):
    """Muestra pequeña: listas largas recortadas a unas filas."""
    if isinstance(raw, dict):
        return {k: _sample(v) for k, v in raw.items()}
    if isinstance(raw, list):
        return [_sample(v) for v in raw[:SAMPLE_ROWS]]
    return raw


def flush(store):
    """Registra lo capturado en este intento. Devuelve {fuente: 'cambio'|'igual'|'error'}."""
    out = {}
    try:
        for source, raw in list(_captured.items()):
            try:
                res = store.record_format(source, shape(raw), _sample(raw))
                out[source] = 'cambio' if res.get('cambio') else 'igual'
            except Exception as exc:
                out[source] = f'error:{type(exc).__name__}'
    finally:
        _captured.clear()
    return out
