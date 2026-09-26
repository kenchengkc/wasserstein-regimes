# SPDX-License-Identifier: GPL-3.0-only
"""Complete inventories and local POSIX publication primitives for new artifacts."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.'+path.name+'-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as handle:
            json.dump(value,handle,sort_keys=True,indent=2,allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def inventory(path):
    root=Path(path)
    files={}
    for item in sorted(root.rglob('*')):
        if item.is_symlink(): raise ValueError('artifact symlinks are unsupported')
        if item.is_file() and item!=root/'checksums.json':
            files[item.relative_to(root).as_posix()]=digest(item)
    return files


def seal(path):
    write_json(Path(path)/'checksums.json',inventory(path))


def verify(path,required=()):
    try:
        expected=json.loads((Path(path)/'checksums.json').read_text())
        if not isinstance(expected,dict) or not expected or not set(required)<=set(expected) or expected!=inventory(path):
            raise ValueError('artifact integrity or inventory mismatch')
    except (OSError,json.JSONDecodeError) as exc:
        raise ValueError('artifact integrity check failed') from exc
    return digest(Path(path)/'checksums.json')


@contextmanager
def lock(path):
    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError('execution requires Linux or macOS advisory locks') from exc
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(handle,fcntl.LOCK_UN)
