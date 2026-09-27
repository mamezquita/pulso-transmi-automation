"""Colector con checkpoints atómicos en Supabase; reiniciable sin SQLite/CSV."""
import argparse
import json
from pathlib import Path
import httpx
from .client import DEFAULT_BASE_URL
from .operational import load_env
from .persistence import RemoteStore
from .contract import ContractError,normalize_observations


QUARANTINE_LIMIT=1000


def _latest_known(store):
    """Referencia para desempatar fechas día/mes: la observación más reciente guardada."""
    try:
        return getattr(store,'latest_observation')('9999-12-31T00:00:00Z')
    except Exception:
        return None


def collect(store,api,max_pages=1000):
    inserted=0;seen=set();conflicts=0;quarantine=[];quarantined=0;formats={}
    near=_latest_known(store)
    for _ in range(max_pages):
        state=store.state();cursor=state['cursor']
        params={'limit':1000}
        if cursor:params['cursor']=cursor
        r=api.get('/v1/stream/observations',params=params);r.raise_for_status()
        page=r.json()
        rows=next((page[k] for k in ('data','items','results') if k in page),None)
        next_cursor=page.get('next_cursor',page.get('nextCursor'))
        if not isinstance(rows,list) or len(rows)>1000:raise ContractError('Página inválida')
        good,bad,fingerprint=normalize_observations(rows,near=near)
        if rows and not good:
            # Cambio de formato sistemático: no avanzar el cursor para no perder datos.
            raise ContractError(f'Página completa en cuarentena: {bad[0]["reason"]}')
        quarantined+=len(bad);quarantine+=bad[:QUARANTINE_LIMIT-len(quarantine)]
        for k,v in fingerprint.items():formats[k]=sorted(set(formats.get(k,[]))|set(v))
        if next_cursor and (next_cursor==cursor or next_cursor in seen):raise ValueError('Cursor repetido')
        try:
            # En la última página conservar el cursor de entrada: se relee sin duplicar.
            result=store.save_page(state['revision'],good,next_cursor or cursor)
        except RuntimeError as exc:
            if 'stale_collector_revision' not in str(exc) or conflicts>=3:raise
            conflicts+=1;seen.clear();continue
        inserted+=result['inserted']
        if good:near=good[-1]['observed_at']
        if next_cursor is None:
            return {'status':'collected','inserted':inserted,'revision':result['revision'],'cursor_saved':True,
                    'quarantined':quarantined,'quarantine':quarantine,'formats':formats}
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
