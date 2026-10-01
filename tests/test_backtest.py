import numpy as np
import pandas as pd
import pytest
from pulso_transmi import backtest as bt

IDX = pd.date_range('2026-09-01', periods=96 * 4, freq='15min', tz='UTC')


def frames(actual_fn, model_fn):
    stations = ['A', 'B']
    actual = pd.DataFrame({s: [actual_fn(i, s) for i in range(len(IDX))] for s in stations}, index=IDX, dtype=float)
    model = pd.DataFrame({s: [model_fn(i, s) for i in range(len(IDX))] for s in stations}, index=IDX, dtype=float)
    return actual, model


def test_bias_correction_recovers_constant_multiplicative_bias():
    actual, model = frames(lambda i, s: 200 + 50 * np.sin(i / 8), lambda i, s: (200 + 50 * np.sin(i / 8)) * 2)
    report = bt.run(actual, model, IDX[100])
    assert report['overall']['modelo_actual']['mae'] > 100
    assert report['overall']['sesgo']['mae'] < 1


def test_blend_prefers_last_value_when_series_is_flat_but_model_is_wrong():
    actual, model = frames(lambda i, s: 300.0, lambda i, s: 100 + (i % 7) * 30)
    report = bt.run(actual, model, IDX[100])
    assert all(w == 1.0 for w in report['params']['mezcla_pesos'].values())
    assert report['overall']['mezcla']['mae'] == 0


def test_predictions_never_use_values_after_the_cutoff():
    actual, model = frames(lambda i, s: 100 + i % 13, lambda i, s: 110.0)
    origin = IDX[200]
    before = bt.simulate(actual, model, [origin], window=12)
    changed = actual.copy()
    changed.loc[changed.index > origin] = 10_000
    after = bt.simulate(changed, model, [origin], window=12)
    cols = ['model', 'last', 'corrected']
    pd.testing.assert_frame_equal(before[cols], after[cols])


def test_origins_skip_incomplete_cycles_and_requires_enough_data():
    actual, model = frames(lambda i, s: 100.0, lambda i, s: 100.0)
    actual.iloc[200:210] = np.nan
    origins = bt.origins_for(actual, IDX[0])
    assert all(o.minute == 0 for o in origins)
    assert not any(IDX[195] <= o <= IDX[209] for o in origins)
    with pytest.raises(ValueError):
        bt.run(actual, model, IDX[-20])
