import httpx
import pandas as pd

from pulso_transmi.persistence import RemoteStore


def test_observations_window_reads_every_page():
    total = 2500
    rows = [{'estacion_id': f'{i % 12:05d}', 'observado_en': f'2026-09-14T{i // 12 % 24:02d}:00:00Z', 'demanda': i}
            for i in range(total)]
    seen = []
    def handler(request):
        params = dict(request.url.params.multi_items())
        offset, limit = int(params['offset']), int(params['limit'])
        seen.append(offset)
        assert params['order'] == 'observado_en.asc,estacion_id.asc'
        return httpx.Response(200, json=rows[offset:offset + limit])
    client = httpx.Client(base_url='https://example.test', transport=httpx.MockTransport(handler))
    store = RemoteStore(url='https://example.test', key='k', client=client)
    got = store.observations_window(pd.Timestamp('2026-09-13', tz='UTC'), pd.Timestamp('2026-09-15', tz='UTC'))
    assert len(got) == total and seen == [0, 1000, 2000]
