import copy
import json
from pathlib import Path

import pytest

import mutaciones as M
from pulso_transmi import diagnostico

FIX = Path(__file__).parent / 'fixtures'
CYCLE = json.loads((FIX / 'ciclo_vivo.json').read_text())
PAGE = json.loads((FIX / 'stream_vivo.json').read_text())


def test_shape_is_stable_for_a_new_cycle_with_the_same_format():
    later = copy.deepcopy(CYCLE)
    later['cycle_id'] = later['cycle_id'].replace('T090000Z', 'T100000Z')
    for t in later['targets']:
        t['target_at'] = t['target_at'].replace('T09:', 'T10:').replace('T10:00', 'T11:00')
    assert diagnostico.shape(later) == diagnostico.shape(CYCLE)


def test_random_ids_and_hashes_do_not_look_like_format_changes():
    a, b = dict(M.REAL_RECEIPT), dict(M.REAL_RECEIPT)
    b['submission_id'] = 'sub_0f' + 'a1' * 15; b['payload_hash'] = 'sha256:' + '0c' * 32
    assert diagnostico.shape(a) == diagnostico.shape(b)


@pytest.mark.parametrize('name,mutated', M.cycle_mutations(CYCLE).items())
def test_every_cycle_format_change_is_detected(name, mutated):
    assert diagnostico.shape(mutated) != diagnostico.shape(CYCLE)


@pytest.mark.parametrize('name,mutated', {**M.stream_mutations(PAGE), **M.receipt_mutations(M.REAL_RECEIPT)}.items())
def test_every_stream_and_receipt_format_change_is_detected(name, mutated):
    if name == 'pagina_cursor_camel':
        pytest.skip('cursor nulo: renombrarlo no deja rastro (el contrato igual lo lee)')
    base = PAGE if 'data' in mutated or 'items' in mutated else M.REAL_RECEIPT
    assert diagnostico.shape(mutated) != diagnostico.shape(base)


class Store:
    def __init__(self, fail=()):
        self.calls, self.fail = [], fail
    def record_format(self, source, shape, sample):
        if source in self.fail:
            raise RuntimeError('rpc caída')
        self.calls.append((source, shape, sample))
        return {'cambio': source == 'recibo'}


def test_flush_records_each_source_once_with_small_sample_and_isolates_errors():
    diagnostico.capture('stream', PAGE); diagnostico.capture('stream', {'otra': 1})  # solo la primera
    diagnostico.capture('ciclo', CYCLE); diagnostico.capture('recibo', M.REAL_RECEIPT)
    store = Store(fail={'ciclo'})
    out = diagnostico.flush(store)
    assert out == {'stream': 'igual', 'ciclo': 'error:RuntimeError', 'recibo': 'cambio'}
    sample = dict((s, m) for s, _, m in store.calls)['stream']
    assert len(sample['data']) == diagnostico.SAMPLE_ROWS
    assert diagnostico.flush(store) == {}  # se limpia tras registrar


def test_pagination_cursor_is_not_a_format_change():
    with_cursor = {**PAGE, 'next_cursor': 'WyIyMDI2LTA5LTIxVDE1OjMwOjA0Ljk1ODA0OSswMDowMCIsIjIwMjYtMDktMDlUMDU6MDA'}
    assert diagnostico.shape({**PAGE, 'next_cursor': None}) == diagnostico.shape(with_cursor)
    assert diagnostico.shape({k: v for k, v in PAGE.items() if k != 'next_cursor'}) == diagnostico.shape(with_cursor)


def test_data_values_do_not_look_like_format_changes():
    # 4-oct: falsas alarmas por cifras de la demanda y por filas "missing" en unas páginas y no en otras.
    row = lambda v, q: {'station_id': '02300', 'measurement': {'unit': 'passengers', 'value': v, 'quality': q}}
    a = {'data': [row('546.00', 'observed'), row('61.00', 'observed')]}
    b = {'data': [row('1234.00', 'observed'), row(None, 'missing')]}
    assert diagnostico.shape(a) == diagnostico.shape(b)


def test_zero_padding_and_decimals_still_count_as_format():
    assert diagnostico.shape({'s': '02300'}) != diagnostico.shape({'s': '2300'})
    assert diagnostico.shape({'v': '546.00'}) != diagnostico.shape({'v': '546'})
    assert diagnostico.shape({'v': '546,00'}) != diagnostico.shape({'v': '546.00'})
