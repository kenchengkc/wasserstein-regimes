# SPDX-License-Identifier: GPL-3.0-only
"""Frozen, exploratory synchronized-panel study; assessment never fits."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import html
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score, silhouette_score
from threadpoolctl import threadpool_limits

from .data import load_csv
from .evaluation import moving_block_indices, temporal_summary
from .experiments import artifact_id, code_provenance, seal_artifact, split_masks, verify_artifact, write_json
from .joint import joint_windows
from .panel_baselines import PanelKMeans, PanelGaussianHMM, panel_features
from .sliced import SlicedWassersteinKMedoids, sliced_w2
from .transport import _positive_int

_FIELDS={'schema_version','assets','provider','price_column','train_end','validation_end','assessment_end',
         'window_length','fit_stride','k_candidates','seed','n_projections','candidate_size','n_init',
         'chunk_size','selection_sample','baseline_n_init','hmm_n_init','hmm_max_iter',
         'projection_counts','projection_seeds','bootstrap_repeats','bootstrap_block_windows','exposure'}
_MODELS=('scaled_joint','raw_joint','marginal','covariance','correlation','hmm')


def validate_config(c):
    if not isinstance(c,dict) or set(c)!=_FIELDS or c['schema_version']!=1:
        raise ValueError('invalid joint study schema or unknown keys')
    if c['exposure']!='exploratory_previously_inspected':
        raise ValueError('assessment exposure must be exploratory_previously_inspected')
    for field in ('window_length','fit_stride','n_projections','candidate_size','n_init','chunk_size',
                  'selection_sample','baseline_n_init','hmm_n_init','hmm_max_iter',
                  'bootstrap_repeats','bootstrap_block_windows'):
        _positive_int(c[field],field)
    for field in ('k_candidates','projection_counts','projection_seeds'):
        if not isinstance(c[field],list) or not c[field] or len(set(c[field]))!=len(c[field]):
            raise ValueError(f'invalid {field}')
        for value in c[field]:
            _positive_int(value,field)
    _positive_int(c['seed'],'seed')
    if min(c['k_candidates'])<2 or c['candidate_size']<max(c['k_candidates']):
        raise ValueError('invalid candidate budget')
    if not pd.Timestamp(c['train_end'])<pd.Timestamp(c['validation_end'])<pd.Timestamp(c['assessment_end']):
        raise ValueError('invalid chronological boundaries')
    assets=c['assets']
    if not isinstance(assets,list) or len(assets)<2:
        raise ValueError('at least two assets required')
    symbols=[]
    for item in assets:
        if set(item)!={'symbol','dataset','sha256'} or not all(isinstance(v,str) and v for v in item.values()):
            raise ValueError('invalid asset specification')
        if len(item['sha256'])!=64 or any(ch not in '0123456789abcdef' for ch in item['sha256']):
            raise ValueError('invalid snapshot hash')
        symbols.append(item['symbol'])
    if len(set(symbols))!=len(symbols):
        raise ValueError('duplicate symbols')


def _crop(series,start,end):
    mask=np.asarray((series.dates>=start)&(series.dates<=end))
    return replace(series,returns=series.returns[mask],dates=series.dates[mask],
                   available_at=series.available_at[mask],price_start=series.price_start[mask],
                   price_end=series.price_end[mask])


def prepare_panel(series,c,cutoff):
    start=max(value.dates[0] for value in series.values())
    end=min(pd.Timestamp(cutoff),min(value.dates[-1] for value in series.values()))
    if start>end:
        raise ValueError('no common history')
    panel={symbol:_crop(value,start,end) for symbol,value in series.items()}
    batch=joint_windows(panel,length=c['window_length'])
    masks=split_masks(batch.dates,batch.price_start,train_end=c['train_end'],
                      validation_end=c['validation_end'],test_end=c['assessment_end'])
    if not masks['train'].any() or not masks['validation'].any():
        raise ValueError('training and validation windows required')
    return panel,batch,masks


def training_scales(panel,train_end):
    scales=np.array([value.returns[value.dates<=train_end].std(ddof=0) for value in panel.values()])
    if not np.isfinite(scales).all() or np.any(scales<=0):
        raise ValueError('invalid unique-return training scales')
    return scales


def load_panel(c,cutoff):
    series,provenance={},{}
    for item in c['assets']:
        path=Path(item['dataset'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('snapshot hash mismatch')
        acquisition=json.loads(path.with_suffix('.json').read_text())
        if acquisition.get('symbol')!=item['symbol'] or acquisition.get('sha256')!=item['sha256']:
            raise ValueError('acquisition metadata mismatch')
        value=load_csv(path,provider=c['provider'],price_column=c['price_column'])
        series[item['symbol']]=value
        provenance[item['symbol']]=dict(acquisition=acquisition,quality=dict(value.quality))
    panel,batch,masks=prepare_panel(series,c,cutoff)
    # The HMM's transition step is daily. Do not bridge missing vector observations.
    first=next(iter(panel.values()))
    if (any(not np.isfinite(value.returns).all() for value in panel.values())
            or not first.price_start[1:].equals(first.price_end[:-1])):
        raise ValueError('daily panel must have complete contiguous return intervals for HMM')
    return panel,batch,masks,provenance


def _directions(dimension,count,seed):
    directions=np.random.default_rng(seed).normal(size=(count,dimension))
    return directions/np.linalg.norm(directions,axis=1,keepdims=True)


def _model(c,k,scales,projections):
    return SlicedWassersteinKMedoids(n_clusters=k,n_projections=len(projections),
        candidate_size=c['candidate_size'],n_init=c['n_init'],chunk_size=c['chunk_size'],
        random_state=c['seed'],scales=scales,projections=projections)


def source_identity():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob('*.py'))}


def _fit_models(c,panel,batch,masks,folder):
    train=batch.samples[masks['train']][::c['fit_stride']]
    validation=batch.samples[masks['validation']]
    scales=training_scales(panel,c['train_end'])
    projections=_directions(train.shape[2],c['n_projections'],c['seed'])
    indices=np.sort(np.random.default_rng(c['seed']).choice(
        len(validation),size=min(len(validation),c['selection_sample']),replace=False))
    anchor=validation[indices]
    distances=sliced_w2(anchor,anchor,projections,scales=scales,chunk_size=c['chunk_size'])
    np.fill_diagonal(distances,0)
    candidates=[]
    selection=[]
    for k in sorted(c['k_candidates']):
        print(f'joint selection K={k}',flush=True)
        model=_model(c,k,scales,projections).fit(train)
        labels=model.predict(anchor)
        occupied=len(np.unique(labels))
        score=float(silhouette_score(distances,labels,metric='precomputed')) if 1<occupied<len(labels) else None
        selection.append(dict(k=k,silhouette=score,occupied=occupied))
        if score is not None:
            candidates.append((score,-k,model))
    if not candidates:
        raise ValueError('all validation candidates degenerate; no fallback K')
    model=max(candidates,key=lambda r:r[:2])[2]
    k=model.n_clusters
    models=dict(scaled_joint=model,raw_joint=_model(c,k,np.ones(len(scales)),projections).fit(train))
    for kind in ('marginal','covariance','correlation'):
        print(f'joint baseline {kind}',flush=True)
        models[kind]=PanelKMeans(kind,k,scales=scales,n_init=c['baseline_n_init'],seed=c['seed']).fit(train)
    first=next(iter(panel.values()))
    vectors=np.column_stack([v.returns for v in panel.values()])
    print('joint baseline full covariance HMM',flush=True)
    models['hmm']=PanelGaussianHMM(k,n_init=c['hmm_n_init'],max_iter=c['hmm_max_iter'],seed=c['seed']).fit(
        vectors[first.dates<=c['train_end']])
    folder.mkdir()
    for name,value in models.items():
        value.save(folder/f'{name}.npz')
    return models,dict(k=k,selection=selection,selection_indices=indices.tolist(),scales=scales.tolist(),
                       training_windows=len(train),hmm_diagnostics=models['hmm'].diagnostics_,
                       hmm_selected_loglikelihood=models['hmm'].selected_loglikelihood_)


def _load_models(folder):
    models={}
    for name in _MODELS:
        cls=SlicedWassersteinKMedoids if name.endswith('joint') else PanelGaussianHMM if name=='hmm' else PanelKMeans
        models[name]=cls.load(folder/f'{name}.npz')
    return models


def _sensitivity(c,batch,masks,model):
    train=batch.samples[masks['train']][::c['fit_stride']]
    validation=batch.samples[masks['validation']]
    reference=model.predict(validation)
    def diagnostic(labels):
        a,b=len(np.unique(reference)),len(np.unique(labels))
        return dict(ari=float(adjusted_rand_score(reference,labels)),occupied=b,
                    reference_occupied=a,both_single_state=a==b==1)
    projections=[]
    for count in c['projection_counts']:
        for seed in c['projection_seeds']:
            print(f'projection sensitivity R={count}, seed={seed}',flush=True)
            fit=_model(c,model.n_clusters,model.scales_,_directions(train.shape[2],count,seed)).fit(train)
            projections.append(dict(projections=count,seed=seed,**diagnostic(fit.predict(validation))))
    bootstrap=[]
    rng=np.random.default_rng(c['seed']+100)
    for repeat in range(c['bootstrap_repeats']):
        indices=moving_block_indices(len(train),block_length=c['bootstrap_block_windows'],rng=rng)
        fit=_model(c,model.n_clusters,model.scales_,model.projections_).fit(train[indices])
        bootstrap.append(dict(repeat=repeat,**diagnostic(fit.predict(validation))))
    return dict(projections=projections,window_block_bootstrap=bootstrap,
                interpretation='Validation only; fixed K/scales; not confidence intervals')


def _summaries(models,panel,batch,mask,k,calibration):
    samples=batch.samples[mask]
    if not len(samples):
        raise ValueError('no strict evaluation windows')
    positions=np.flatnonzero(mask)
    first=next(iter(panel.values()))
    vectors=np.column_stack([v.returns for v in panel.values()])
    hmm=models['hmm'].predict(vectors)
    endpoints=first.dates.get_indexer(batch.dates[mask])
    if np.any(endpoints<0):
        raise ValueError('missing HMM endpoint')
    assignments={name:(hmm[endpoints] if name=='hmm' else model.predict(samples)) for name,model in models.items()}
    vol=samples.std(axis=1,ddof=0).mean(axis=1)
    corr=panel_features(samples,'correlation').mean(axis=1)
    metrics={}
    reference=assignments['scaled_joint']
    for name,labels in assignments.items():
        counts=np.bincount(labels,minlength=k)
        both=len(np.unique(reference))==len(np.unique(labels))==1
        row=dict(n=len(labels),counts=counts.tolist(),occupancy=(counts/len(labels)).tolist(),
                 occupied=int(np.count_nonzero(counts)),ari_vs_scaled_joint=float(adjusted_rand_score(reference,labels)),
                 both_single_state_vs_scaled=both,temporal=temporal_summary(labels,positions,k=k),
                 contemporaneous=[dict(state=state,n=int(counts[state]),
                    mean_asset_daily_volatility=float(vol[labels==state].mean()) if counts[state] else None,
                    mean_pair_correlation=float(corr[labels==state].mean()) if counts[state] else None) for state in range(k)])
        if name.endswith('joint'):
            nearest=models[name].transform(samples).min(axis=1)
            threshold=calibration[name]
            row.update(novelty_threshold=threshold,novelty_fraction=float(np.mean(nearest>threshold)))
        metrics[name]=row
    frame=pd.DataFrame(dict(date=batch.dates[mask],price_start=batch.price_start[mask],
                            price_end=batch.price_end[mask],available_at=batch.available_at[mask],**assignments))
    return metrics,frame


def _render(folder):
    metrics=json.loads((folder/'metrics.json').read_text())
    rows=[dict(model=name,occupied=value['occupied'],windows=value['n'],
               ARI_vs_scaled_joint=value['ari_vs_scaled_joint'],
               both_single_state=value['both_single_state_vs_scaled'],counts=str(value['counts']))
          for name,value in metrics['models'].items()]
    page='<!doctype html><html lang="en"><meta charset="utf-8"><title>Exploratory joint market study</title>'
    page+='<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px}td,th{padding:8px}pre{white-space:pre-wrap}</style>'
    page+='<h1>Exploratory joint market study</h1><p>Previously inspected market period; no confirmatory or predictive claim. All models score common strict windows. Single-state agreement is flagged.</p>'
    page+=pd.DataFrame(rows).to_html(index=False,escape=True)
    page+='<details><summary>Complete aggregate diagnostics</summary><pre>'+html.escape(json.dumps(metrics,indent=2))+'</pre></details></html>'
    (folder/'report.html').write_text(page)
    return folder/'report.html'


def run_joint_market(config_path,*,output_root='artifacts',stage='development',development=None):
    if stage not in ('development','assessment'):
        raise ValueError('invalid study stage')
    if stage=='assessment' and development is None:
        raise ValueError('verified development artifact required before assessment')
    c=yaml.safe_load(Path(config_path).read_text())
    validate_config(c)
    provenance=code_provenance()
    sources=source_identity()
    if stage=='assessment':
        dev=Path(development)
        verify_artifact(dev)
        prior=json.loads((dev/'config.json').read_text())
        manifest=json.loads((dev/'manifest.json').read_text())
        if prior!=c or manifest['source_files']!=sources or manifest['stage']!='development':
            raise ValueError('development protocol or source differs from assessment')
        if manifest['dependency_versions']!=provenance['dependency_versions']:
            raise ValueError('development dependencies differ from assessment')
        target=dev.parent/'assessment'
    else:
        identity=artifact_id(c,dict(source_files=sources,dependencies=provenance['dependency_versions']))
        target=Path(output_root)/f'joint-market-{identity}'/'development'
    cutoff=c['validation_end'] if stage=='development' else c['assessment_end']
    panel,batch,masks,data=load_panel(c,cutoff)
    if (target/'checksums.json').exists():
        verify_artifact(target)
        return _render(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix=stage+'-partial-',dir=target.parent))
    with threadpool_limits(limits=1):
        if stage=='development':
            models,fit=_fit_models(c,panel,batch,masks,temp/'models')
            # Model reload is part of execution, not just a serialization unit test.
            models=_load_models(temp/'models')
            calibration={name:float(np.quantile(models[name].transform(batch.samples[masks['validation']]).min(axis=1),.99))
                         for name in ('scaled_joint','raw_joint')}
            sensitivity=_sensitivity(c,batch,masks,models['scaled_joint'])
            selected=masks['validation']
        else:
            dev_metrics=json.loads((dev/'metrics.json').read_text())
            fit,calibration=dev_metrics['fit'],dev_metrics['calibration']
            models=_load_models(dev/'models')
            sensitivity=None
            selected=masks['test']
        metrics,assignments=_summaries(models,panel,batch,selected,fit['k'],calibration)
    write_json(temp/'config.json',c)
    first=next(iter(panel.values()))
    manifest=dict(stage=stage,exposure=c['exposure'],symbol_order=list(panel),
                  common_first_return=str(first.dates[0].date()),common_first_price=str(first.price_start[0].date()),
                  common_last_return=str(first.dates[-1].date()),return_count=len(first.returns),
                  source_files=sources,dependency_versions=provenance['dependency_versions'],
                  execution_provenance=provenance,data=data,
                  development=None if stage=='development' else str(dev))
    write_json(temp/'manifest.json',manifest)
    write_json(temp/'metrics.json',dict(stage=stage,fit=fit,models=metrics,sensitivity=sensitivity,calibration=calibration))
    assignments.to_parquet(temp/'assignments.parquet',index=False)
    seal_artifact(temp)
    temp.rename(target)
    return _render(target)
