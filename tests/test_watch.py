import pytest
from pulso_transmi.watch import watch


class Timer:
    now = 0
    def clock(self): return self.now
    def sleep(self, seconds):
        assert 0 < seconds <= 60
        self.now += seconds


def test_keeps_polling_after_closed_window_and_acceptance():
    timer = Timer()
    results = iter([
        {'forecast': {'status': 'no_open_cycle', 'submitted': False}},
        {'forecast': {'status': 'accepted', 'submitted': True}},
        {'forecast': {'status': 'already_accepted', 'submitted': False}},
    ])
    reports = []
    result = watch(lambda: next(results), reports.append, minutes=5,
                   clock=timer.clock, sleep=timer.sleep)
    assert result == {'status': 'finished', 'attempts': 3, 'new_submissions': 1}
    assert len(reports) == 3 and timer.now == 300


def test_transient_exception_is_sanitized_and_retried():
    timer = Timer()
    reports = []
    def attempt():
        if not reports:
            raise ConnectionError('secret text')
        return {'forecast': {'submitted': False}}
    result = watch(attempt, reports.append, minutes=3, clock=timer.clock, sleep=timer.sleep)
    assert result['status'] == 'finished' and result['attempts'] == 2
    assert reports[0] == {'status': 'error', 'error_type': 'ConnectionError'}


def test_repeated_errors_stop_instead_of_consuming_whole_session():
    timer = Timer()
    result = watch(lambda: {'status': 'error'}, lambda r: None, minutes=300,
                   clock=timer.clock, sleep=timer.sleep)
    assert result['status'] == 'failed' and result['attempts'] == 5
    assert timer.now < 600


def test_cancel_stops_before_next_attempt():
    timer = Timer()
    results = []
    result = watch(lambda: {}, results.append, minutes=300,
                   stopped=lambda: bool(results), clock=timer.clock, sleep=timer.sleep)
    assert result['status'] == 'interrupted' and result['attempts'] == 1


@pytest.mark.parametrize('minutes', [0, -1, 301])
def test_duration_cannot_be_unbounded(minutes):
    with pytest.raises(ValueError):
        watch(lambda: {}, lambda r: None, minutes=minutes)
