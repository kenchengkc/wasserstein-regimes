# SPDX-License-Identifier: GPL-3.0-only
"""One process per attempt; OS lock and atomic result pointer protect resumption."""
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
import uuid

from .artifact_store import digest,lock,seal,verify,write_json
from .jobs import verify_request


def rss_bytes(raw,platform_name):
    if platform_name=='darwin': return int(raw)
    if platform_name=='linux': return int(raw)*1024
    raise RuntimeError('RSS units unsupported on this platform')


def cached_result(directory,request):
    pointer=directory/'result.json'
    if not pointer.exists(): return None
    saved=json.loads(pointer.read_text())
    if set(saved)!={'attempt','digest','job_id'} or saved['job_id']!=request['id']:
        raise ValueError('invalid job result pointer')
    name=saved['attempt']
    if not isinstance(name,str) or len(name)!=32 or any(c not in '0123456789abcdef' for c in name):
        raise ValueError('invalid attempt identifier')
    target=directory/'attempts'/name
    actual=verify(target,{'status.json','execution.json','task.json'})
    state=json.loads((target/'status.json').read_text())
    if saved['digest']!=actual or state.get('status')!='completed' or state.get('job_id')!=request['id']:
        raise ValueError('job result integrity mismatch')
    return target


def execute(request,directory):
    directory=Path(directory)
    verify_request(request)
    with lock(directory/'job.lock'):
        verify_request(request)
        result=cached_result(directory,request)
        if result is not None: return dict(status='cached',output=str(result))
        attempts=directory/'attempts'
        attempts.mkdir(exist_ok=True)
        for previous in attempts.iterdir():
            state=previous/'status.json'
            if state.exists() and json.loads(state.read_text()).get('status')=='running':
                write_json(state,dict(status='interrupted',job_id=request['id'],reason='previous owner released lock without publishing'))
        attempt=attempts/uuid.uuid4().hex
        attempt.mkdir()
        write_json(attempt/'status.json',dict(status='running',job_id=request['id'],pid=os.getpid()))
        started=time.perf_counter()
        before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        try:
            from threadpoolctl import threadpool_limits,threadpool_info
            from .job_tasks import dispatch
            with threadpool_limits(limits=request['identity']['blas_threads']):
                task=dispatch(request['identity']['spec'],attempt)
                pools=threadpool_info()
                if any(p['num_threads']>request['identity']['blas_threads'] for p in pools):
                    raise RuntimeError('observed numerical threads exceed requested limit')
            verify_request(request)
            peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            write_json(attempt/'task.json',task)
            write_json(attempt/'execution.json',dict(pid=os.getpid(),platform=platform.platform(),python=platform.python_version(),
                elapsed_seconds=time.perf_counter()-started,peak_rss_bytes=rss_bytes(peak,sys.platform),
                pre_task_peak_rss_bytes=rss_bytes(before,sys.platform),ru_maxrss_raw=peak,
                ru_maxrss_unit='bytes' if sys.platform=='darwin' else 'KiB',
                memory_scope='process lifetime high-water RSS through task completion, includes imports and input arrays',
                blas_threads=request['identity']['blas_threads'],threadpools=pools))
            write_json(attempt/'status.json',dict(status='completed',job_id=request['id'],pid=os.getpid()))
            seal(attempt)
            checksum=verify(attempt,{'status.json','execution.json','task.json'})
            write_json(directory/'result.json',dict(attempt=attempt.name,digest=checksum,job_id=request['id']))
            return dict(status='completed',output=str(attempt))
        except BaseException as exc:
            write_json(attempt/'status.json',dict(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                job_id=request['id'],error=type(exc).__name__,message=str(exc)))
            (attempt/'failure.txt').write_text(traceback.format_exc())
            raise


def main():
    directory=Path(sys.argv[1]).resolve()
    request=json.loads((directory/'request.json').read_text())
    print(json.dumps(execute(request,directory)),flush=True)


if __name__=='__main__': main()
