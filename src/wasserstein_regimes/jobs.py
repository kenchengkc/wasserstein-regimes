# SPDX-License-Identifier: GPL-3.0-only
"""Content-bound local jobs with bounded fresh-process execution."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import uuid

import yaml

from .artifact_store import digest,identity,lock,write_json

_SOURCE_NAMES=('__init__.py','artifact_store.py','frozen.py','jobs.py','job_worker.py','job_tasks.py',
    'sliced.py','transport.py','joint.py','joint_market.py','joint_validation.py','joint_controls.py',
    'robustness.py','panel_baselines.py','data.py','windows.py','evaluation.py','experiments.py','core.py','clustering.py')
_PACKAGES=('numpy','scipy','scikit-learn','pandas','pyarrow','hmmlearn','exchange-calendars','matplotlib','PyYAML','threadpoolctl')
_FIELDS={'benchmark':{'n_windows','length','dimensions','k','projections','candidates','n_init','seed'},
         'score':{'bundle','config'},'refit':{'development','market_config','train_end','seed','candidates'}}


def _integer(value,name,minimum=1,maximum=None):
    if type(value) is not int or value<minimum or (maximum is not None and value>maximum):
        raise ValueError(f'invalid {name}')


def runtime_identity():
    versions={}
    for package in _PACKAGES:
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]='unavailable'
    return dict(sources={n:digest(Path(__file__).parent/n) for n in _SOURCE_NAMES},
                dependencies=versions,python=list(sys.version_info[:2]),platform=sys.platform)


def _input_files(spec):
    files=[]
    if spec['type']=='benchmark': return {}
    from .frozen import load_bundle,validate_score_config,session_date
    if spec['type']=='score':
        root=Path(spec['bundle'])
        contract,_=load_bundle(root)
        cfg=Path(spec['config'])
        config=yaml.safe_load(cfg.read_text())
        validate_score_config(config,contract)
        files += [root/'model.npz',root/'contract.json',root/'checksums.json',cfg]
    else:
        from .experiments import verify_artifact
        from .joint_market import validate_config
        root=Path(spec['development'])
        verify_artifact(root)
        inventory=json.loads((root/'checksums.json').read_text())
        if not {'config.json','manifest.json','metrics.json','models/scaled_joint.npz'}<=set(inventory):
            raise ValueError('incomplete development inventory')
        files += [root/'checksums.json']+[root/name for name in inventory]
        cfg=Path(spec['market_config'])
        config=yaml.safe_load(cfg.read_text())
        validate_config(config)
        if session_date(spec['train_end'])>session_date(config['train_end']):
            raise ValueError('refit cannot extend beyond training')
        files.append(cfg)
    for asset in config['assets']:
        path=Path(asset['dataset']).absolute()
        if digest(path)!=asset['sha256']: raise ValueError('snapshot hash mismatch')
        files.extend([path,path.with_suffix('.json')])
    # Keep lexical paths: execution derives sidecars beside the configured CSV,
    # and verification must observe later symlink retargeting.
    return {str(p.absolute()):digest(p) for p in files}


def prepare_job(job,*,blas_threads=1):
    _integer(blas_threads,'blas_threads',maximum=8)
    if not isinstance(job,dict) or job.get('type') not in _FIELDS:
        raise ValueError('unknown job type')
    kind=job['type']
    if set(job)!=_FIELDS[kind]|{'name','type'} or not isinstance(job['name'],str) or not job['name']:
        raise ValueError('invalid job schema')
    spec=copy.deepcopy(job)
    name=spec.pop('name')
    if kind=='benchmark':
        for key in _FIELDS[kind]: _integer(spec[key],key,minimum=0 if key=='seed' else 1)
        if spec['candidates']<spec['k'] or spec['n_windows']<spec['k']:
            raise ValueError('candidate or sample budget below K')
    elif kind=='refit':
        _integer(spec['seed'],'seed',minimum=0)
        _integer(spec['candidates'],'candidates',minimum=2)
    for key in ('bundle','development','config','market_config'):
        if key in spec:
            if not isinstance(spec[key],str) or not spec[key]: raise ValueError('invalid input path')
            spec[key]=str(Path(spec[key]).resolve())
    computation=dict(spec=spec,inputs=_input_files(spec),blas_threads=blas_threads,cwd=str(Path.cwd()),**runtime_identity())
    return dict(id=identity(computation),name=name,identity=computation)


def verify_request(request):
    value=request['identity']
    if identity(value)!=request['id']: raise ValueError('job identity mismatch')
    if str(Path.cwd())!=value['cwd']: raise ValueError('job working directory differs')
    current=runtime_identity()
    if any(value[k]!=v for k,v in current.items()): raise ValueError('job source or dependency identity changed')
    if any(not Path(path).is_file() or digest(path)!=expected for path,expected in value['inputs'].items()):
        raise ValueError('job input identity changed')


def _stop(process):
    if process.poll() is not None: return
    try: os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError: return
    try: process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try: os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait()


def run_jobs(jobs,*,output_root='artifacts/jobs',workers=2,blas_threads=1,timeout=3600):
    _integer(workers,'workers',maximum=8)
    _integer(blas_threads,'blas_threads',maximum=8)
    if sys.platform not in ('linux','darwin'): raise RuntimeError('jobs require Linux or macOS')
    if type(timeout) not in (int,float) or not 0<timeout<float('inf'): raise ValueError('invalid timeout')
    if not isinstance(jobs,list) or not jobs: raise ValueError('nonempty jobs required')
    requests=[prepare_job(j,blas_threads=blas_threads) for j in jobs]
    if len({r['id'] for r in requests})!=len(requests): raise ValueError('duplicate computational jobs')
    if len({r['name'] for r in requests})!=len(requests): raise ValueError('duplicate job names')
    root=Path(output_root).resolve()
    run=root/'runs'/uuid.uuid4().hex
    run.mkdir(parents=True)
    for request in requests:
        directory=root/request['id']
        directory.mkdir(parents=True,exist_ok=True)
        with lock(directory/'request.lock'):
            path=directory/'request.json'
            if path.exists():
                old=json.loads(path.read_text())
                if old['identity']!=request['identity'] or old['id']!=request['id']:
                    raise ValueError('existing request differs')
            else: write_json(path,request)
    active={}
    guard=threading.Lock()
    stopping=threading.Event()
    def launch(request):
        directory=root/request['id']
        log=run/(request['id']+'.log')
        env=os.environ.copy()
        for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
            env[key]=str(blas_threads)
        # Use this checkout even if the active interpreter has another editable install.
        source=str(Path(__file__).resolve().parents[1])
        env['PYTHONPATH']=source+os.pathsep+env.get('PYTHONPATH','')
        with log.open('w') as stream:
            with guard:
                if stopping.is_set(): return dict(name=request['name'],id=request['id'],status='cancelled')
                proc=subprocess.Popen([sys.executable,'-m','wasserstein_regimes.job_worker',str(directory)],
                    cwd=request['identity']['cwd'],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                active[request['id']]=proc
            timed_out=False
            try:
                try: code=proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out=True
                    _stop(proc)
                    code=proc.returncode
            finally:
                with guard: active.pop(request['id'],None)
        row=dict(name=request['name'],id=request['id'],log=str(log),returncode=code,timed_out=timed_out)
        if code!=0 or timed_out: return dict(row,status='failed')
        from .job_worker import cached_result
        result=cached_result(directory,request)
        if result is None: return dict(row,status='failed',error='worker exited without sealed result')
        # Worker writes its cache/completion outcome only to its own log.
        event=json.loads(log.read_text().splitlines()[-1])
        return dict(row,status=event['status'],output=str(result))
    pool=ThreadPoolExecutor(max_workers=workers)
    rows=[]
    try:
        futures=[pool.submit(launch,r) for r in requests]
        for future in as_completed(futures):
            row=future.result()
            rows.append(row)
            print(f"{row['name']}: {row['status']}",flush=True)
    except BaseException:
        stopping.set()
        with guard: processes=list(active.values())
        for proc in processes: _stop(proc)
        raise
    finally: pool.shutdown(wait=True,cancel_futures=True)
    order={r['id']:i for i,r in enumerate(requests)}
    rows.sort(key=lambda r:order[r['id']])
    success=all(r['status'] in ('completed','cached') for r in rows)
    path=run/'summary.json'
    write_json(path,dict(schema_version=1,success=success,workers=workers,blas_threads=blas_threads,jobs=rows))
    if not success: raise RuntimeError(f'jobs failed; inspect {path}')
    return path


def run_job_file(path,**kwargs):
    config=yaml.safe_load(Path(path).read_text())
    if not isinstance(config,dict) or set(config)!={'schema_version','jobs'} or type(config['schema_version']) is not int or config['schema_version']!=1:
        raise ValueError('invalid batch schema')
    return run_jobs(config['jobs'],**kwargs)
