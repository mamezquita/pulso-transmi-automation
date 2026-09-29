import numpy as np
import pandas as pd

from pulso_transmi.shadow import lookback, ratio_conjunto, ratio_estacional

CUTOFF = pd.Timestamp('2026-09-14T15:00:00Z')
TARGETS = pd.DataFrame({'station_id': ['A', 'A'], 'target_at': ['2026-09-14T15:15:00Z', '2026-09-14T16:00:00Z']})


class Model:
    def predict(self, df): return np.full(len(df), 100.0)


def history(values_by_time):
    rows = [{'station_id': 'A', 'observed_at': pd.Timestamp(t), 'demand': v} for t, v in values_by_time.items()]
    return pd.DataFrame(rows, columns=['station_id', 'observed_at', 'demand'])


def last_hour(values):  # 4 obs terminando en el corte, de la más antigua a la más reciente
    times = pd.date_range(CUTOFF - 3 * pd.Timedelta(minutes=15), CUTOFF, freq='15min')
    return dict(zip(times, values))


def test_conjunto_averages_ratios_of_each_window():
    h = history(last_hour([100, 100, 200, 200]))  # r2 = 400/200 = 2.0 ; r4 = 600/400 = 1.5
    values, diag = ratio_conjunto(np.array([100.0, 100.0]), TARGETS, h, Model(), CUTOFF, ventanas=(2, 4))
    assert diag['ratios'] == {'A': 1.75} and list(values) == [175.0, 175.0]


def test_conjunto_uses_available_windows_and_falls_back_to_model():
    h = history({t: v for t, v in last_hour([0, 0, 150, 150]).items() if v})  # faltan las 2 más antiguas
    values, diag = ratio_conjunto(np.array([100.0, 100.0]), TARGETS, h, Model(), CUTOFF, ventanas=(2, 4))
    assert diag['ratios'] == {'A': 1.5}  # solo la ventana de 2 está completa
    values, diag = ratio_conjunto(np.array([100.0, 100.0]), TARGETS, history({}), Model(), CUTOFF)
    assert list(values) == [100.0, 100.0] and diag['sin_datos'] == ['A']


def test_estacional_mixes_yesterday_scaled_by_today_level():
    day = pd.Timedelta(days=1)
    data = last_hour([100, 100, 120, 120])  # r2 = 240/200 = 1.2 ; nivel en el corte = 120
    data[CUTOFF - day] = 60                  # ayer en el corte: hoy/ayer = 2.0
    data[pd.Timestamp('2026-09-14T15:15:00Z') - day] = 80
    data[pd.Timestamp('2026-09-14T16:00:00Z') - day] = 50
    values, diag = ratio_estacional(np.array([100.0, 100.0]), TARGETS, history(data), Model(), CUTOFF,
                                    ventana_obs=2, peso=0.2)
    assert list(values) == [round(0.8 * 120 + 0.2 * 160), round(0.8 * 120 + 0.2 * 100)]
    assert diag['estacional_usada'] == 2


def test_estacional_without_yesterday_is_plain_ratio_and_ignores_future():
    data = last_hour([100, 100, 120, 120])
    data[pd.Timestamp('2026-09-14T15:15:00Z')] = 10_000  # posterior al corte: nunca debe usarse
    h = history(data)
    h = h[h.observed_at <= CUTOFF]  # load_history ya filtra; aquí se replica esa barrera
    values, diag = ratio_estacional(np.array([100.0, 100.0]), TARGETS, h, Model(), CUTOFF, ventana_obs=2)
    assert list(values) == [120.0, 120.0] and diag['estacional_usada'] == 0


def test_lookback_covers_every_parameter_style():
    assert lookback({'ventana_obs': 4}) == 4
    assert lookback({'ventanas': [2, 4]}) == 4
    assert lookback({'ventana_obs': 2, 'historial_obs': 100}) == 100
    assert lookback({}) == 0
