import copy
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from pulso_transmi.collector import collect
from pulso_transmi.forecaster import forecast,prepare
from pulso_transmi.persistence import RemoteStore

CYCLE={'cycle_id':'cyc_test','state':'open','data_cutoff':'2026-09-11T16:00:00Z',
       'closes_at':'2099-01-01T00:00:00Z','expected_predictions':1,
       'targets':[{'station_id':'00001','target_at':'2026-09-11T16:15:00Z'}]}
RECEIPT={'submission_id':'sub_test','status':'accepted','predictions_received':1}

class FakeStore:
    def __init__(self):
        self.record=None;self.loads=0;self.fail_finish=False
        self.meta={'sha256':'a'*64,'version':'calendar-test','metadata':{'training_data_end':'2026-09-10T00:00:00Z','trained_at':'2026-09-23T00:00:00Z'}}
    def job(self,c):return copy.deepcopy(self.record)
    def active_model(self):return self.meta
    def latest_observation(self,c):return CYCLE['data_cutoff']
    def load_model(self,r):
        self.loads+=1
        return SimpleNamespace(end=r['metadata']['training_data_end'],predict=lambda d:np.array([42.]))
    def reserve(self,p,sha):
        if self.record is None:self.record={'ciclo_id':p['cycle_id'],'client_run_id':p['client_run_id'],'payload':copy.deepcopy(p),'respuesta':None}
        return copy.deepcopy(self.record)
    def finish(self,job,response=None,error=None):
        if self.fail_finish:self.fail_finish=False;raise RuntimeError('Database offline after POST')
        self.record['respuesta']=response


def api_client(sent):
    def handler(r):
        if r.method=='GET':return httpx.Response(200,json=CYCLE)
        sent.append((r.headers['Idempotency-Key'],json.loads(r.content)))
        return httpx.Response(201,json=RECEIPT)
    return httpx.Client(base_url='https://pulso-transmi.72-60-245-2.sslip.io',transport=httpx.MockTransport(handler))


def test_fresh_runner_recovers_accepted_without_loading_model_or_post():
    store=FakeStore();prepare(store,CYCLE)
    store.record['respuesta']={'http_status':201,'response':RECEIPT};store.loads=0
    with api_client([]) as api:r=forecast(store,api,submit=True,token='test')
    assert r['status']=='already_accepted' and store.loads==0


def test_prepare_only_and_inference_reuses_active_model():
    store=FakeStore();sent=[]
    with api_client(sent) as api:
        first=forecast(store,api)
        second=forecast(store,api)
    assert first==second and store.loads==1 and sent==[]


def test_post_succeeded_but_receipt_write_failed_retry_uses_identical_key_and_payload():
    store=FakeStore();store.fail_finish=True;sent=[]
    with api_client(sent) as api:
        with pytest.raises(RuntimeError):forecast(store,api,submit=True,token='test')
        result=forecast(store,api,submit=True,token='test')
        again=forecast(store,api,submit=True,token='test')
    assert result['status']=='accepted' and again['status']=='already_accepted'
    assert len(sent)==2 and sent[0]==sent[1] and store.loads==1


def test_winner_payload_used_in_concurrent_reservation():
    store=FakeStore();original=store.reserve
    def reserve(p,sha):
        other=copy.deepcopy(p);other['client_run_id']='winner-run';other['predictions'][0]['value']=84
        return original(other,sha)
    store.reserve=reserve;sent=[]
    with api_client(sent) as api:forecast(store,api,submit=True,token='test')
    assert sent[0][0]=='winner-run' and sent[0][1]['predictions'][0]['value']==84


def test_future_trained_model_rejected_before_loading():
    store=FakeStore();store.meta['metadata']['training_data_end']='2027-01-01T00:00:00Z'
    with pytest.raises(ValueError,match='después'):prepare(store,CYCLE)
    assert store.loads==0


def test_closed_window_never_posts():
    store=FakeStore();sent=[];closed=copy.deepcopy(CYCLE);closed['closes_at']='2000-01-01T00:00:00Z'
    with httpx.Client(base_url='https://pulso-transmi.72-60-245-2.sslip.io',transport=httpx.MockTransport(lambda r:(sent.append(r.method) or httpx.Response(200,json=closed)))) as api:
        assert forecast(store,api,submit=True,token='test')['status']=='cycle_closed'
    assert set(sent)=={'GET'}


def test_remote_model_checksum_checked_before_deserialization():
    with httpx.Client(base_url='https://example.test',transport=httpx.MockTransport(lambda r:httpx.Response(200,content=b'corrupt'))) as client:
        store=RemoteStore('https://example.test','test',client)
        with pytest.raises(ValueError,match='SHA-256'):
            store.load_model({'metadata':{'versions':{}},'storage_path':'model.joblib','sha256':'a'*64})


def test_collector_atomic_checkpoint_and_restart():
    class State:
        data={'cursor':None,'revision':0};keys=set()
        def state(self):return dict(self.data)
        def save_page(self,rev,rows,cursor):
            assert rev==self.data['revision']
            old=len(self.keys);self.keys.update(r['observed_at'] for r in rows)
            self.data={'revision':rev+1,'cursor':cursor}
            return {**self.data,'inserted':len(self.keys)-old}
    store=State();calls=[]
    def handler(r):
        cur=r.url.params.get('cursor');calls.append(cur)
        row={'station_id':'00001','observed_at':'2026-09-11T16:00:00Z' if cur is None else '2026-09-11T16:15:00Z','demand':1,'released_at':'2026-09-24T00:00:00Z'}
        return httpx.Response(200,json={'data':[row],'next_cursor':'page2' if cur is None else None})
    with httpx.Client(base_url='https://example.test',transport=httpx.MockTransport(handler)) as api:
        assert collect(store,api)['inserted']==2
        assert collect(store,api)['inserted']==0
    assert calls==[None,'page2','page2'] and store.data['cursor']=='page2'


def test_collector_save_failure_does_not_advance_state():
    store=SimpleNamespace(state=lambda:{'revision':0,'cursor':None})
    def fail(*a):raise RuntimeError('transaction rolled back')
    store.save_page=fail
    with httpx.Client(base_url='https://example.test',transport=httpx.MockTransport(lambda r:httpx.Response(200,json={'data':[],'next_cursor':'page2'}))) as api:
        with pytest.raises(RuntimeError):collect(store,api)
    assert store.state()['cursor'] is None


def test_calendar_artifact_loads_in_fresh_process_without_main_alias(tmp_path):
    import subprocess,sys,joblib
    from pulso_transmi.operational import CalendarModel
    dates=__import__('pandas').date_range('2026-09-01',periods=8,freq='15min',tz='UTC')
    frame=__import__('pandas').DataFrame({'station_id':'00001','observed_at':dates,'demand':np.arange(8)+1})
    model=CalendarModel().fit(frame)
    path=tmp_path/'model.joblib';joblib.dump(model,path)
    result=subprocess.run([sys.executable,'-c',
        "import joblib,sys; m=joblib.load(sys.argv[1]); assert type(m).__module__=='pulso_transmi.operational'; print('portable')",str(path)],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='portable'
