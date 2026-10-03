"""Cada cambio de formato del banco de mutaciones debe dar el mismo resultado normalizado
que la respuesta real original (ciclo, página del stream y recibo del envío)."""
import copy
import json
from pathlib import Path

import pandas as pd
import pytest

import mutaciones as M
from pulso_transmi.contract import (ContractError, normalize_cycle_contract, normalize_observations, normalize_receipt,
                                    page_parts)

FIX = Path(__file__).parent / 'fixtures'
CYCLE = json.loads((FIX / 'ciclo_vivo.json').read_text())
PAGE = json.loads((FIX / 'stream_vivo.json').read_text())


def cycle_key(c):
    close = pd.Timestamp(c['closes_at']).floor('min') if c.get('closes_at') else None
    return (c['cycle_id'], c['data_cutoff'], c['state'], close, c['expected_predictions'],
            sorted((t['station_id'], t['target_at'], t.get('horizon_minutes')) for t in c['targets']))


def page_rows(page):
    return page_parts(page)[0]


def trimmed(rows):  # algunas mutaciones pierden los segundos de released_at
    return [{**r, 'released_at': r['released_at'][:16]} for r in rows]


BASE_CYCLE = cycle_key(normalize_cycle_contract(CYCLE))
BASE_ROWS = trimmed(normalize_observations(PAGE['data'])[0])


@pytest.mark.parametrize('name,mutated', M.cycle_mutations(CYCLE).items())
def test_cycle_survives_format_change(name, mutated):
    assert cycle_key(normalize_cycle_contract(mutated)) == BASE_CYCLE


@pytest.mark.parametrize('name,mutated', M.stream_mutations(PAGE).items())
def test_stream_survives_format_change(name, mutated):
    good, bad, _ = normalize_observations(page_rows(mutated))
    assert bad == [] and trimmed(good) == BASE_ROWS


@pytest.mark.parametrize('name,mutated', M.receipt_mutations(M.REAL_RECEIPT).items())
def test_receipt_survives_format_change(name, mutated):
    r = normalize_receipt(mutated, 201)
    assert r['status'] == 'accepted' and r['submission_id'] == M.REAL_RECEIPT['submission_id']


def test_receipt_rejection_or_http_error_is_never_turned_into_acceptance():
    assert normalize_receipt({**M.REAL_RECEIPT, 'status': 'rejected'}, 201)['status'] == 'rejected'
    assert normalize_receipt({**M.REAL_RECEIPT, 'status': 'no'}, 201)['status'] == 'no'
    assert normalize_receipt({**M.REAL_RECEIPT, 'status': 'accepted'}, 422)['status'] == 'accepted'  # sin cambios
    assert normalize_receipt({**M.REAL_RECEIPT, 'status': 'yes'}, 500)['status'] == 'yes'


def test_unreadable_target_dates_are_rebuilt_from_cycle_id_and_horizons():
    c = copy.deepcopy(CYCLE)
    for t in c['targets']:
        t['target_at'] = 'pronto'
    for k in ('origin_at', 'forecast_start_at', 'forecast_end_at', 'data_cutoff'):
        c[k] = 'hora secreta'
    out = normalize_cycle_contract(c)
    assert cycle_key(out) == BASE_CYCLE and 'reconstruido' in out


def test_rebuild_refuses_when_readable_cutoff_contradicts_cycle_id():
    c = copy.deepcopy(CYCLE)
    c['data_cutoff'] = '2026-09-20T07:00:00Z'
    for t in c['targets']:
        t['target_at'] = 'pronto'
    with pytest.raises(ContractError):
        normalize_cycle_contract(c)


def test_rebuild_needs_every_horizon():
    c = copy.deepcopy(CYCLE)
    for t in c['targets']:
        t['target_at'] = 'pronto'
    del c['targets'][0]['horizon_minutes']
    with pytest.raises(ContractError):
        normalize_cycle_contract(c)


def test_unreadable_closing_time_is_left_empty_not_invented():
    c = copy.deepcopy(CYCLE)
    c['closes_at'] = 'en un rato'
    out = normalize_cycle_contract(c)
    assert out['closes_at'] is None and out['formato_no_leido'] == ['closes_at']
    assert out['targets'] == normalize_cycle_contract(CYCLE)['targets']


def test_unknown_columns_that_are_ambiguous_go_to_quarantine_instead_of_guessing():
    # Dos columnas enteras con la misma cantidad de valores distintos: no se sabe cuál es la estación.
    rows = [{'a': 2300 + i, 'b': 100 + i, 'ts': '2026-09-09T05:00:00Z', 'pub': '2026-09-21T15:30:00Z'} for i in range(4)]
    good, bad, _ = normalize_observations(rows)
    assert good == [] and len(bad) == 4


def test_known_field_names_never_trigger_content_inference(monkeypatch):
    import pulso_transmi.contract as contract
    monkeypatch.setattr(contract, '_infer_columns', lambda rows: (_ for _ in ()).throw(AssertionError('no debía inferir')))
    assert len(normalize_observations(PAGE['data'])[0]) == len(PAGE['data'])
