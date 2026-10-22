"""Loopback-only browser app for the local rocket research lab."""
import csv
import io
import json
import os
import re
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field

from .telemetry import ROOT, DATA, DB, flight_rows, read_flight
from .evaluation import landing_interval

@asynccontextmanager
async def lifespan(app):
    yield
    with job_lock:
        proc = job['process']
        if proc is not None:
            job['state'] = 'stopped'
            proc.terminate()
    if proc is not None:
        proc.wait(timeout=10)


app=FastAPI(title='Rocket Landing Lab',docs_url=None,redoc_url=None,lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware,allowed_hosts=['127.0.0.1','localhost','testserver'])
app.mount('/assets',StaticFiles(directory=ROOT/'rocket_landing/web'),name='assets')
app.mount('/docs',StaticFiles(directory=ROOT/'docs'),name='reports')
job={'state':'idle','lines':[],'kind':None,'process':None}
job_lock=threading.Lock()


@app.middleware('http')
async def local_requests(request:Request,call_next):
    if request.method=='POST':
        origin=request.headers.get('origin')
        if origin and origin.rstrip('/')!=str(request.base_url).rstrip('/'):
            return Response('Only requests from this local app are accepted.',status_code=403)
    response=await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff'
    return response


@app.get('/')
@app.get('/lab')
def page():return FileResponse(ROOT/'rocket_landing/web/index.html')


def filtered(model='',scenario='',outcome=''):
    return [r for r in flight_rows() if (not model or r['model']==model) and
            (not scenario or r['scenario']==scenario) and (not outcome or r['outcome']==outcome)]


@app.get('/api/flights')
def flights(model:str='',scenario:str='',outcome:str=''):
    return filtered(model,scenario,outcome)


@app.get('/api/summary')
def summary(model:str='',scenario:str='',outcome:str=''):
    rows=filtered(model,scenario,outcome)
    groups=[]
    for name in sorted({r['model'] for r in rows}):
        selected=[r for r in rows if r['model']==name]
        n=len(selected);k=sum(r['outcome']=='landed' for r in selected)
        groups.append({'model':name,'episodes':n,'landed':k/n,'ci':landing_interval(k,n),
                       'score':sum(r['score'] for r in selected)/n,'fuel':sum(r['fuel'] for r in selected)/n,
                       'timeout':sum(r['outcome']=='timeout' for r in selected)/n})
    return {'count':len(rows),'models':groups,'landed':sum(r['outcome']=='landed' for r in rows),
            'mean_score':sum(r['score'] for r in rows)/len(rows) if rows else 0,
            'mean_seconds':sum(r['steps']/50 for r in rows)/len(rows) if rows else 0,
            'seed_range':[min(r['seed'] for r in rows),max(r['seed'] for r in rows)] if rows else [],
            'outcomes':{o:sum(r['outcome']==o for r in rows) for o in ['landed','failed','timeout']}}


@app.get('/api/flights/{identity}')
def flight(identity:str):
    if not re.fullmatch('[0-9a-f]{20}',identity):raise HTTPException(404)
    record=read_flight(identity)
    if not record:raise HTTPException(404)
    from .predictor import predict_flight
    record['forecast']=predict_flight(record)
    return record


@app.get('/api/export')
def export(model:str='',scenario:str='',outcome:str=''):
    rows=filtered(model,scenario,outcome)
    stream=io.StringIO()
    writer=csv.DictWriter(stream,fieldnames=['id','model','scenario','seed','model_sha256','outcome','score','steps','fuel','hover'])
    writer.writeheader();writer.writerows(rows)
    return Response(stream.getvalue(),media_type='text/csv',headers={'Content-Disposition':'attachment; filename=rocket-flights.csv'})


@app.get('/api/predictor')
def predictor():
    path=DATA/'predictor-report.json'
    if not path.exists():return {'available':False}
    return {'available':True,**json.loads(path.read_text())}


@app.get('/api/training')
def training():
    path=DATA/'broader-results.json'
    return json.loads(path.read_text()) if path.exists() else {'cases':[],'protocol':'No completed broader-weather evaluation yet.'}


@app.get('/api/assistant/status')
def assistant_status():
    from .assistant import ollama_status
    return ollama_status()


class Question(BaseModel):
    question:str=Field(min_length=3,max_length=1000)


@app.post('/api/ask')
def ask(payload:Question):
    from .assistant import answer
    return answer(payload.question)


@app.get('/api/demo/{name}')
def demo(name:str):
    if name not in ['baseline','fixed','hover','time']:raise HTTPException(404)
    candidates=sorted((ROOT/f'runs/{name}/seed_0/videos').glob('*/*.gif'))
    if not candidates:raise HTTPException(404)
    return FileResponse(candidates[0])


@app.get('/api/report/{kind}')
def report(kind:str):
    if kind=='original':path=ROOT/'artifacts/report.html'
    elif kind=='quality':
        candidates=sorted((ROOT/'artifacts/quality').glob('*/quality-gate.html'))
        if not candidates:raise HTTPException(404)
        path=candidates[-1]
    else:raise HTTPException(404)
    # Reports use relative media links; direct users to the PDF for portable context.
    return FileResponse(ROOT/'docs/Rocket-Landing-Placement-Report.pdf')


class JobRequest(BaseModel):
    kind:str
    episodes:int=Field(default=60,ge=10,le=200)
    seed:int=Field(default=600000,ge=0,le=2000000000)
    steps:int=Field(default=200000,ge=10000,le=1000000)


@app.post('/api/jobs')
def start_job(payload:JobRequest):
    with job_lock:
        if job['process'] is not None:raise HTTPException(409,'A lab task is already running or stopping. Wait for completion.')
        if payload.kind=='collect':args=['rocket_landing.telemetry','--episodes',str(payload.episodes),'--seed',str(payload.seed)]
        elif payload.kind=='predictor':args=['rocket_landing.predictor']
        elif payload.kind=='train':
            seeds=[int(p.name.split('_')[-1]) for p in (ROOT/'runs/robust').glob('seed_*') if p.name.split('_')[-1].isdigit()]
            args=['rocket_landing.broader_training','--steps',str(payload.steps),'--seed',str(max(seeds,default=0)+1)]
        else:raise HTTPException(400,'Unknown task')
        proc=subprocess.Popen([sys.executable,'-u','-m',*args],cwd=ROOT,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',
                              env=dict(os.environ,PYTHONUTF8='1'),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        job.update(state='running',lines=[],kind=payload.kind,process=proc)
        def receive():
            for line in proc.stdout:
                with job_lock:job['lines']=(job['lines']+[line.rstrip()])[-200:]
            proc.stdout.close();code=proc.wait()
            with job_lock:
                if job['state']!='stopped':job['state']='complete' if code==0 else 'failed'
                job['process']=None
        threading.Thread(target=receive,daemon=True).start()
    return {'state':'running'}


@app.get('/api/jobs')
def job_status():
    with job_lock:return {k:v for k,v in job.items() if k!='process'}


@app.post('/api/jobs/stop')
def stop_job():
    with job_lock:
        if job['process'] is not None:
            job['state']='stopped';job['process'].terminate()
    return {'state':'stopped'}


def main():
    import argparse
    import uvicorn
    import socket
    import webbrowser
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=0)
    parser.add_argument('--open-browser',action='store_true')
    args=parser.parse_args()
    sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    sock.bind(('127.0.0.1',args.port))
    port=sock.getsockname()[1]
    if args.open_browser:
        threading.Timer(1.5,lambda:webbrowser.open(f'http://127.0.0.1:{port}')).start()
    print(f'Open Rocket Landing Lab: http://127.0.0.1:{port}',flush=True)
    try:uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port)).run(sockets=[sock])
    finally:sock.close()


if __name__=='__main__':main()
