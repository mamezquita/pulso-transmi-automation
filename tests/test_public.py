from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import pytest
from pulso_transmi.automation import save_report

spec = importlib.util.spec_from_file_location('session_gate', Path(__file__).parents[1]/'scripts/session_gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_public_report_does_not_publish_private_fields(tmp_path):
    path = tmp_path/'report.json'
    save_report({'participant': 'PRIVATE_NAME', 'attempt_id': 'PRIVATE_ID',
        'forecast': {'status': 'accepted', 'submitted': True,
                     'receipt': {'response': {'submission_id': 'PRIVATE_RECEIPT'}}}}, path)
    text = path.read_text()
    assert 'PRIVATE' not in text
    assert json.loads(text)['submitted']


def test_campaign_deadline():
    now = datetime(2026, 9, 26, tzinfo=timezone.utc)
    assert gate.allowed_minutes('2026-10-10T00:00:00Z',300,now) == 300
    assert gate.allowed_minutes('2026-09-26T00:02:59Z',300,now) == 2
    assert gate.allowed_minutes('2026-09-25T00:00:00Z',300,now) == 0
    with pytest.raises(ValueError): gate.allowed_minutes('2026-10-10',300,now)
    with pytest.raises(ValueError): gate.allowed_minutes('2026-10-10T00:00:00Z',301,now)
