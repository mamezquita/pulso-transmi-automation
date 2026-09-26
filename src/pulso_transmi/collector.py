"""Colector con checkpoints atómicos en Supabase; reiniciable sin SQLite/CSV."""
import argparse
import json
import re
from pathlib import Path
import httpx
from .client import DEFAULT_BASE_URL
from .operational import load_env
from .persistence import RemoteStore
from .prepare_submission import timestamp


def collect(store,api,max_pages=1000):
    inserted=0;seen=set();conflicts=0
    for _ in range(max_pages):
        state=store.state();cursor=state['cursor']
        params={'limit':1000}
        if cursor:params['cursor']=cursor
        r=api.get('/v1/stream/observations',params=params);r.raise_for_status()
        page=r.json();rows=page['data'];next_cursor=page.get('next_cursor')
        if not isinstance(rows,list) or len(rows)>1000:raise ValueError('Página inválida')
        for row in rows:
            if not isinstance(row['station_id'],str) or not re.fullmatch('[0-9]{5}',row['station_id']):raise ValueError('Estación inválida')
            if type(row['demand']) is not int or row['demand']<0:raise ValueError('Demanda inválida')
            if timestamp(row['observed_at']).value%900_000_000_000:raise ValueError('Timestamp desalineado')
            timestamp(row['released_at'])
        if next_cursor and (next_cursor==cursor or next_cursor in seen):raise ValueError('Cursor repetido')
        try:
            # En la última página conservar el cursor de entrada: se relee sin duplicar.
            result=store.save_page(state['revision'],rows,next_cursor or cursor)
        except RuntimeError as exc:
            if 'stale_collector_revision' not in str(exc) or conflicts>=3:raise
            conflicts+=1;seen.clear();continue
        inserted+=result['inserted']
        if next_cursor is None:
            return {'status':'collected','inserted':inserted,'revision':result['revision'],'cursor_saved':True}
        seen.add(next_cursor)
    raise ValueError('Se alcanzó el límite de páginas; checkpoint guardado, reanudar otra ejecución')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file',type=Path,default=Path('.env'))
    args=parser.parse_args();load_env(args.env_file)
    store=RemoteStore()
    try:
        with httpx.Client(base_url=DEFAULT_BASE_URL,timeout=60) as api:print(json.dumps(collect(store,api)))
    finally:store.close()

if __name__=='__main__':main()
