from datetime import datetime, timezone

from pulso_transmi.reentreno import decide, safe_now

POL = {'umbral_cambio_modelo': 80, 'margen_cambio_pp': 2}
r = lambda fam, acc: {'familia': fam, 'prueba': {'accuracy': acc}}


def test_trees_kept_when_they_reach_80_and_beat_delivery():
    d = decide(POL, {'accuracy': 67}, r('lightgbm', 81), [])
    assert d[0] == 'nueva_revision' and d[1] == 'lightgbm' and d[3] == 'arboles'


def test_trees_at_80_but_not_better_than_delivery_changes_nothing():
    assert decide(POL, {'accuracy': 85}, r('lightgbm', 82), [])[0] == 'sin_cambio'


def test_other_model_only_if_it_beats_trees_by_the_margin():
    others = [r('bosque_aleatorio', 80.5), r('knn', 70)]
    d = decide(POL, {'accuracy': 67}, r('lightgbm', 78), others)
    assert d[0] == 'nueva_version' and d[1] == 'bosque_aleatorio'
    d = decide(POL, {'accuracy': 67}, r('lightgbm', 78), [r('bosque_aleatorio', 79.9)])
    assert d[0] == 'nueva_revision' and d[1] == 'lightgbm'  # +1.9 pp no alcanza el margen


def test_nothing_changes_if_no_candidate_beats_delivery():
    assert decide(POL, {'accuracy': 79}, r('lightgbm', 75), [r('knn', 76)])[0] == 'sin_cambio'


class Store:
    def __init__(self, last_cycle, new_code=True): self.last, self.new_code = last_cycle, new_code
    def rows(self, table, **k):
        if table == 'intento_operativo':
            return [{'resultado': {'etapas': []} if self.new_code else {}}]
        return [{'ciclo_id': self.last, 'respuesta': {}}] if self.last else []


def test_activation_only_in_safe_minutes_and_after_this_hour_delivery():
    at = lambda m: datetime(2026, 10, 1, 3, m, tzinfo=timezone.utc)
    assert not safe_now(Store('c1'), at(5), None)[0]
    assert safe_now(Store(None), at(15), None)[0]                       # sin ciclo abierto
    assert safe_now(Store('c1'), at(15), {'cycle_id': 'c1'})[0]         # ya entregado
    assert not safe_now(Store('c0'), at(15), {'cycle_id': 'c1'})[0]     # abierto y sin entregar
    assert not safe_now(Store('c1'), at(35), None)[0]


def test_activation_waits_until_delivery_sessions_run_the_new_code():
    at = datetime(2026, 10, 1, 3, 15, tzinfo=timezone.utc)
    ok, reason = safe_now(Store(None, new_code=False), at, None)
    assert not ok and 'código anterior' in reason
    assert safe_now(Store(None, new_code=True), at, None)[0]
