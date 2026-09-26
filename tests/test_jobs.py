# SPDX-License-Identifier: GPL-3.0-only
import copy
import json
import os
from pathlib import Path

import pytest


def benchmark(name='small',seed=42):
    return dict(name=name,type='benchmark',n_windows=32,length=5,dimensions=2,k=2,projections=4,candidates=8,n_init=1,seed=seed)


def test_job_schema_and_content_identity(tmp_path):
    from wasserstein_regimes.jobs import prepare_job
    a=prepare_job(benchmark(),blas_threads=1)
    assert a['id']==prepare_job(benchmark(name='changed display label'),blas_threads=1)['id']
    assert a['id']!=prepare_job(benchmark(seed=83),blas_threads=1)['id']
    assert a['id']!=prepare_job(benchmark(),blas_threads=2)['id']
    assert 'reporting.py' not in a['identity']['sources']
    assert not any(name.endswith('.md') for name in a['identity']['sources'])
    for changes in [dict(extra=1),dict(seed=True),dict(n_windows=0),dict(candidates=1),dict(type='shell')]:
        with pytest.raises(ValueError): prepare_job(dict(benchmark(),**changes),blas_threads=1)


def test_real_workers_isolated_and_second_batch_reused(tmp_path):
    from wasserstein_regimes.jobs import run_jobs
    first=run_jobs([benchmark(),benchmark('other',83)],output_root=tmp_path,workers=2,blas_threads=1,timeout=30)
    one=json.loads(first.read_text())
    assert one['success'] and all(r['status']=='completed' for r in one['jobs'])
    metrics=[json.loads((Path(r['output'])/'execution.json').read_text()) for r in one['jobs']]
    assert len({m['pid'] for m in metrics})==2 and all(m['pid']!=os.getpid() for m in metrics)
    assert all(m['peak_rss_bytes']>0 and m['blas_threads']==1 for m in metrics)
    assert all(p['num_threads']<=1 for m in metrics for p in m['threadpools'])
    second=run_jobs([benchmark(),benchmark('other',83)],output_root=tmp_path,workers=1,blas_threads=1,timeout=30)
    two=json.loads(second.read_text())
    assert all(r['status']=='cached' for r in two['jobs'])
    assert [r['output'] for r in one['jobs']]==[r['output'] for r in two['jobs']]
    output=Path(one['jobs'][0]['output'])
    (output/'task.json').write_text('{}')
    with pytest.raises(RuntimeError,match='failed'):
        run_jobs([benchmark()],output_root=tmp_path,workers=1,blas_threads=1,timeout=30)


def test_interrupted_attempt_retried_without_false_success(tmp_path,monkeypatch):
    from wasserstein_regimes.jobs import prepare_job
    from wasserstein_regimes.job_worker import execute
    import wasserstein_regimes.job_tasks as tasks
    request=prepare_job(benchmark(),blas_threads=1)
    original=tasks.dispatch
    monkeypatch.setattr(tasks,'dispatch',lambda *a,**k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt): execute(request,tmp_path)
    assert not (tmp_path/'result.json').exists()
    monkeypatch.setattr(tasks,'dispatch',original)
    outcome=execute(request,tmp_path)
    assert outcome['status']=='completed'
    attempts=list((tmp_path/'attempts').iterdir())
    assert len(attempts)==2
    states=[json.loads((p/'status.json').read_text())['status'] for p in attempts]
    assert sorted(states)==['completed','interrupted']


def test_rss_units_and_source_recheck(tmp_path):
    from wasserstein_regimes.jobs import prepare_job
    from wasserstein_regimes.job_worker import execute,rss_bytes
    assert rss_bytes(1024,'darwin')==1024
    assert rss_bytes(1024,'linux')==1048576
    with pytest.raises(RuntimeError): rss_bytes(1024,'win32')
    request=prepare_job(benchmark(),blas_threads=1)
    request['identity']['sources']['sliced.py']='changed'
    with pytest.raises(ValueError,match='identity'): execute(request,tmp_path)


def test_timeout_attempt_can_resume(tmp_path):
    from wasserstein_regimes.jobs import run_jobs
    with pytest.raises(RuntimeError,match='failed'):
        run_jobs([benchmark()],output_root=tmp_path,workers=1,blas_threads=1,timeout=.001)
    result=run_jobs([benchmark()],output_root=tmp_path,workers=1,blas_threads=1,timeout=30)
    assert json.loads(result.read_text())['success']


def test_concurrent_coordinators_publish_one_attempt(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from wasserstein_regimes.jobs import run_jobs
    def run(name):
        return json.loads(run_jobs([benchmark(name)],output_root=tmp_path,workers=1,blas_threads=1,timeout=30).read_text())
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(run,['first','second']))
    assert sorted(r['jobs'][0]['status'] for r in results)==['cached','completed']
    assert len({r['jobs'][0]['output'] for r in results})==1
    assert len(list((Path(results[0]['jobs'][0]['output']).parent).iterdir()))==1


def test_changed_input_during_execution_cannot_publish(tmp_path,monkeypatch):
    from wasserstein_regimes.jobs import prepare_job
    from wasserstein_regimes.job_worker import execute
    from wasserstein_regimes.artifact_store import digest,identity
    import wasserstein_regimes.job_tasks as tasks
    source=tmp_path/'source.json'
    source.write_text('{}')
    request=prepare_job(benchmark(),blas_threads=1)
    request['identity']['inputs']={str(source):digest(source)}
    request['id']=identity(request['identity'])
    original=tasks.dispatch
    def changed(spec,output):
        result=original(spec,output)
        source.write_text('{"changed":true}')
        return result
    monkeypatch.setattr(tasks,'dispatch',changed)
    with pytest.raises(ValueError,match='input identity'):
        execute(request,tmp_path/'job')
    assert not (tmp_path/'job/result.json').exists()
