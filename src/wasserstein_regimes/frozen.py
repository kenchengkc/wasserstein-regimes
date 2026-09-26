# SPDX-License-Identifier: GPL-3.0-only
"""Immutable joint-model export and contract-checked inference; never fits."""
from dataclasses import replace
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import pandas as pd
import yaml

from .artifact_store import digest,identity,lock,seal,verify,write_json
from .data import load_csv
from .experiments import code_provenance,verify_artifact,_json_safe
from .joint import joint_windows
from .joint_market import check_acquisition
from .sliced import SlicedWassersteinKMedoids

_NUMERICAL=('sliced.py','transport.py','data.py','joint.py','windows.py')
_CONTRACT={'schema_version','symbols','window_length','provider','price_column','return_basis','calendar',
           'train_end','calibration_end','novelty_threshold','model','parent_digest','numerical_sources','dependencies'}
_SCORE={'schema_version','assets','provider','price_column','start','end','as_of'}


def numerical_sources():
    root=Path(__file__).parent
    return {name:digest(root/name) for name in _NUMERICAL}


def session_date(value):
    if not isinstance(value,str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}',value):
        raise ValueError('expected ISO session date')
    return pd.Timestamp(value)


def validate_contract(c):
    if not isinstance(c,dict) or set(c)!=_CONTRACT or type(c['schema_version']) is not int or c['schema_version']!=1:
        raise ValueError('invalid frozen schema')
    if (not isinstance(c['symbols'],list) or not c['symbols'] or any(not isinstance(s,str) or not s for s in c['symbols'])
            or len(set(c['symbols']))!=len(c['symbols']) or type(c['window_length']) is not int or c['window_length']<1):
        raise ValueError('invalid frozen dimensions')
    if c['return_basis']!='adjusted_close' or c['calendar']!='XNYS' or c['model'] not in ('scaled_joint','raw_joint'):
        raise ValueError('unsupported return or model convention')
    if any(not isinstance(c[k],str) or not c[k] for k in ('provider','price_column','parent_digest')):
        raise ValueError('invalid frozen metadata')
    if session_date(c['train_end'])>=session_date(c['calibration_end']):
        raise ValueError('invalid calibration chronology')
    if type(c['novelty_threshold']) not in (float,int) or not np.isfinite(c['novelty_threshold']) or c['novelty_threshold']<0:
        raise ValueError('invalid novelty threshold')
    if c['numerical_sources']!=numerical_sources(): raise ValueError('frozen numerical source mismatch')
    if c['dependencies']!=code_provenance()['dependency_versions']: raise ValueError('frozen dependency mismatch')


def load_bundle(path):
    path=Path(path)
    verify(path,{'contract.json','model.npz'})
    c=json.loads((path/'contract.json').read_text())
    validate_contract(c)
    model=SlicedWassersteinKMedoids.load(path/'model.npz')
    if model.medoids_.shape[1:]!=(c['window_length'],len(c['symbols'])):
        raise ValueError('frozen model dimensions differ from contract')
    return c,model


def freeze_joint(development,*,model='scaled_joint',output_root='artifacts/bundles'):
    if model not in ('scaled_joint','raw_joint'): raise ValueError('unsupported frozen model')
    dev=Path(development)
    verify_artifact(dev)
    inventory=json.loads((dev/'checksums.json').read_text())
    required={'config.json','manifest.json','metrics.json',f'models/{model}.npz'}
    if not required<=set(inventory): raise ValueError('incomplete development inventory')
    config=json.loads((dev/'config.json').read_text())
    manifest=json.loads((dev/'manifest.json').read_text())
    metrics=json.loads((dev/'metrics.json').read_text())
    if manifest['stage']!='development': raise ValueError('development stage required')
    sources=numerical_sources()
    if any(manifest['source_files'].get(n)!=h for n,h in sources.items()):
        raise ValueError('development numerical source mismatch')
    symbols=[a['symbol'] for a in config['assets']]
    if symbols!=manifest['symbol_order']: raise ValueError('development asset order mismatch')
    contract=dict(schema_version=1,symbols=symbols,window_length=config['window_length'],provider=config['provider'],
        price_column=config['price_column'],return_basis='adjusted_close',calendar='XNYS',train_end=config['train_end'],
        calibration_end=config['validation_end'],novelty_threshold=metrics['calibration'][model],model=model,
        parent_digest=digest(dev/'checksums.json'),numerical_sources=sources,dependencies=manifest['dependency_versions'])
    validate_contract(contract)
    model_digest=digest(dev/f'models/{model}.npz')
    name='joint-bundle-'+identity(dict(contract=contract,model_sha256=model_digest))[:24]
    target=Path(output_root)/name
    with lock(Path(output_root)/'.locks'/name):
        if target.exists():
            load_bundle(target)
            return target
        target.parent.mkdir(parents=True,exist_ok=True)
        temp=Path(tempfile.mkdtemp(prefix=name+'-partial-',dir=target.parent))
        shutil.copyfile(dev/f'models/{model}.npz',temp/'model.npz')
        write_json(temp/'contract.json',contract)
        seal(temp)
        load_bundle(temp)
        verify_artifact(dev)
        if digest(dev/'checksums.json')!=contract['parent_digest'] or digest(temp/'model.npz')!=model_digest:
            raise ValueError('parent changed during bundle export')
        temp.rename(target)
    return target


def validate_score_config(c,contract):
    if not isinstance(c,dict) or set(c)!=_SCORE or type(c['schema_version']) is not int or c['schema_version']!=1:
        raise ValueError('invalid score schema')
    if c['provider']!=contract['provider'] or c['price_column']!=contract['price_column']:
        raise ValueError('score provider or price convention differs')
    assets=c['assets']
    if not isinstance(assets,list) or any(not isinstance(a,dict) or set(a)!={'symbol','dataset','sha256'} for a in assets):
        raise ValueError('invalid score assets')
    if [a['symbol'] for a in assets]!=contract['symbols']: raise ValueError('asset order differs')
    for a in assets:
        if any(not isinstance(v,str) or not v for v in a.values()) or not re.fullmatch('[0-9a-f]{64}',a['sha256']):
            raise ValueError('invalid snapshot specification')
    _dates(c['start'],c['end'],c['as_of'])


def _dates(start,end,as_of):
    first,last=session_date(start),session_date(end)
    ready=pd.Timestamp(as_of)
    if pd.isna(ready) or ready.tzinfo is None: raise ValueError('as_of must include timezone')
    if first>last: raise ValueError('scoring start after end')
    return first,last,ready.tz_convert('UTC')


def score_panel(panel,contract,model,*,start,end,as_of):
    validate_contract(contract)
    if list(panel)!=contract['symbols']: raise ValueError('asset order differs')
    if any(s.provider!=contract['provider'] or s.return_basis!=contract['return_basis'] for s in panel.values()):
        raise ValueError('input convention differs')
    first,last,ready=_dates(start,end,as_of)
    common=max(s.dates[0] for s in panel.values())
    finish=min(last,min(s.dates[-1] for s in panel.values()))
    if common>finish: raise ValueError('no common scoring history')
    cropped={}
    for symbol,s in panel.items():
        use=np.asarray((s.dates>=common)&(s.dates<=finish))
        cropped[symbol]=replace(s,returns=s.returns[use],dates=s.dates[use],available_at=s.available_at[use],
                                price_start=s.price_start[use],price_end=s.price_end[use])
    batch=joint_windows(cropped,contract['window_length'])
    eligible=np.asarray((batch.dates>=first)&(batch.dates<=last))
    training=np.asarray(batch.price_start<=contract['train_end'])
    unavailable=np.asarray(batch.available_at>ready)
    selected=eligible&~training&~unavailable
    if not selected.any(): raise ValueError('no complete, available, post-training scoring windows')
    distances=model.transform(batch.samples[selected])
    labels=distances.argmin(axis=1)
    nearest=distances.min(axis=1)
    retrospective=np.asarray(batch.dates[selected]<=contract['calibration_end'])
    frame=pd.DataFrame(dict(date=batch.dates[selected],price_start=batch.price_start[selected],
        price_end=batch.price_end[selected],available_at=batch.available_at[selected],state=labels,
        nearest_distance=nearest,novelty=nearest>contract['novelty_threshold'],calibration_reuse=retrospective))
    summary=dict(windows=len(frame),counts=np.bincount(labels,minlength=model.n_clusters).tolist(),
        novelty_fraction=float(frame.novelty.mean()),excluded_training_overlap=int((eligible&training).sum()),
        excluded_unavailable=int((eligible&~training&unavailable).sum()),
        novelty_interpretation='retrospective_calibration_reuse' if retrospective.any() else 'post_calibration_scoring',
        observation_availability_only=True,point_in_time_snapshot=False,
        period=[str(frame.date.iloc[i].date()) for i in (0,-1)])
    return frame,summary


def score_snapshot(bundle,config_path,output):
    contract,model=load_bundle(bundle)
    c=yaml.safe_load(Path(config_path).read_text())
    validate_score_config(c,contract)
    panel,metadata={},{}
    for a in c['assets']:
        path=Path(a['dataset'])
        if digest(path)!=a['sha256']: raise ValueError('snapshot hash mismatch')
        acquisition=json.loads(path.with_suffix('.json').read_text())
        check_acquisition(acquisition,a['symbol'],a['sha256'])
        value=load_csv(path,provider=c['provider'],price_column=c['price_column'])
        if value.dataset_sha256!=a['sha256']: raise ValueError('snapshot changed while loading')
        panel[a['symbol']]=value
        metadata[a['symbol']]=dict(acquisition=acquisition,quality=_json_safe(dict(value.quality)))
    frame,summary=score_panel(panel,contract,model,start=c['start'],end=c['end'],as_of=c['as_of'])
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    frame.to_parquet(output/'scores.parquet',index=False)
    write_json(output/'scoring.json',dict(summary=summary,config=c,data=metadata,bundle_digest=digest(Path(bundle)/'checksums.json')))
    return summary
