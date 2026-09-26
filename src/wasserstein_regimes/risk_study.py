# SPDX-License-Identifier: GPL-3.0-only
"""Frozen-model exploratory risk study; market audit rows stay in local artifacts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .artifact_store import digest,write_json
from .data import load_csv
from .frozen import load_bundle,session_date,validate_contract,validate_score_config
from .joint import joint_windows
from .joint_market import _crop,check_acquisition
from .risk import METHODS,_integer,causal_forecasts,forecast_losses,paired_comparisons

_SCORE_FIELDS={'schema_version','assets','provider','price_column','start','end','as_of'}
_RISK_FIELDS={'horizon','min_history','ewma_decay','prior_count','forecast_floor','bootstrap_repeats',
              'bootstrap_blocks','bootstrap_seed','exposure'}


def validate_risk_config(c,contract):
    if not isinstance(c,dict) or set(c)!=_SCORE_FIELDS|_RISK_FIELDS:
        raise ValueError('invalid risk schema')
    validate_score_config({k:c[k] for k in _SCORE_FIELDS},contract)
    if session_date(c['start'])<=session_date(contract['calibration_end']):
        raise ValueError('risk assessment must follow model calibration')
    if c['exposure']!='exploratory_previously_inspected': raise ValueError('risk exposure must be exploratory')
    for key in ('horizon','min_history','prior_count','bootstrap_repeats','bootstrap_seed'):
        _integer(c[key],key,0 if key=='bootstrap_seed' else 2 if key=='bootstrap_repeats' else 1)
    if c['min_history']<contract['window_length']: raise ValueError('minimum history below window')
    if type(c['ewma_decay']) not in (int,float) or not 0<c['ewma_decay']<1: raise ValueError('invalid EWMA decay')
    if type(c['forecast_floor']) not in (int,float) or not 0<c['forecast_floor']<np.inf: raise ValueError('invalid forecast floor')
    blocks=c['bootstrap_blocks']
    if not isinstance(blocks,list) or not blocks: raise ValueError('invalid bootstrap blocks')
    for block in blocks: _integer(block,'bootstrap block')
    if len(set(blocks))!=len(blocks): raise ValueError('duplicate bootstrap blocks')


def _panel(panel,c,contract):
    if list(panel)!=contract['symbols']: raise ValueError('asset order differs')
    if any(s.provider!=contract['provider'] or s.return_basis!=contract['return_basis'] for s in panel.values()):
        raise ValueError('input convention differs')
    if any(not len(s.dates) for s in panel.values()): raise ValueError('empty panel')
    start=max(s.dates[0] for s in panel.values())
    end=min(session_date(c['end']),min(s.dates[-1] for s in panel.values()))
    if start>end: raise ValueError('no common risk history')
    panel={name:_crop(s,start,end) for name,s in panel.items()}
    first=next(iter(panel.values()))
    if (any(not np.isfinite(s.returns).all() for s in panel.values())
            or not first.price_start[1:].equals(first.price_end[:-1])):
        raise ValueError('risk requires complete contiguous daily returns')
    window=contract['window_length']
    batch=joint_windows(panel,window)  # validates exact dates, intervals and basis
    if len(batch.samples)!=len(first.returns)-window+1: raise ValueError('incomplete risk windows')
    available=pd.to_datetime(np.column_stack([s.available_at.as_unit('ns').asi8 for s in panel.values()]).max(axis=1),utc=True)
    if not np.all(np.diff(available.asi8)>0): raise ValueError('risk requires strictly increasing availability')
    return panel,first,batch,available


def _metrics(frame):
    losses=forecast_losses(frame.target.to_numpy(),frame[list(METHODS)].to_numpy())
    return {method:dict(origins=len(frame),mean_forecast=float(frame[method].mean()),mean_target=float(frame.target.mean()),
                        **{name:float(values[:,j].mean()) for name,values in losses.items()})
            for j,method in enumerate(METHODS)}


def risk_panel(panel,c,contract,model):
    validate_contract(contract)
    validate_risk_config(c,contract)
    panel,first,batch,available=_panel(panel,c,contract)
    window=contract['window_length']
    n=len(first.returns)
    h=c['horizon']
    log_returns=np.column_stack([s.returns for s in panel.values()])
    with np.errstate(over='ignore',invalid='ignore'):
        q=np.mean(np.expm1(log_returns),axis=1)**2
    if not np.isfinite(q).all(): raise ValueError('nonfinite basket squared returns')
    trailing=np.array([q[t-window+1:t+1].mean() for t in range(window-1,n)])
    training=np.asarray(batch.dates<=contract['train_end'])
    if not training.any(): raise ValueError('no training history for volatility boundaries')
    boundaries=np.quantile(trailing[training],[1/3,2/3])
    labels=np.full(n,-1,dtype=int)
    labels[window-1:]=model.predict(batch.samples)
    vol_labels=np.full(n,-1,dtype=int)
    vol_labels[window-1:]=np.searchsorted(boundaries,trailing,side='right')
    result=causal_forecasts(q,labels,vol_labels,k=model.n_clusters,window=window,
                           **{k:c[k] for k in ('horizon','min_history','prior_count','ewma_decay','forecast_floor')})
    ends=np.arange(window-1,n)
    requested=np.asarray((batch.dates>=c['start'])&(batch.dates<=c['end']))
    post_calibration=np.asarray(batch.price_start>contract['calibration_end'])
    ready=pd.Timestamp(c['as_of']).tz_convert('UTC')
    is_available=np.asarray(batch.available_at<=ready)
    warmed=ends>=c['min_history']-1
    selected=requested&post_calibration&is_available&warmed
    positions=ends[selected]
    if not len(positions): raise ValueError('no post-calibration risk origins')
    frame=pd.DataFrame(dict(date=first.dates[positions],price_start=batch.price_start[selected],
        available_at=available[positions],state=labels[positions],volatility_state_label=vol_labels[positions],
        regime_count=result['regime_count'][positions],volatility_count=result['volatility_count'][positions]))
    for j,method in enumerate(METHODS): frame[method]=result['forecasts'][positions,j]
    targets=np.full(len(positions),np.nan)
    maturities=[];target_available=[]
    for j,t in enumerate(positions):
        if t+h<n:
            maturities.append(first.dates[t+h])
            target_available.append(available[t+h])
            if available[t+h]<=ready: targets[j]=float(q[t+1:t+h+1].mean())
        else:
            maturities.append(pd.NaT);target_available.append(pd.NaT)
    frame['target']=targets
    frame['target_end']=pd.to_datetime(maturities)
    frame['target_available_at']=pd.to_datetime(target_available,utc=True)
    frame['scored']=np.isfinite(targets)
    scored=frame[frame.scored]
    if len(scored)<=max(c['bootstrap_blocks']): raise ValueError('too few scored origins for bootstrap blocks')
    # Strict daily prefixes guarantee scored origins are contiguous; never compact gaps.
    if not np.array_equal(np.flatnonzero(frame.scored),np.arange(len(scored))):
        raise ValueError('noncontiguous assessment targets')
    losses=forecast_losses(scored.target.to_numpy(),scored[list(METHODS)].to_numpy())
    for loss,values in losses.items():
        for j,method in enumerate(METHODS):
            frame.loc[frame.scored,f'{loss}_{method}']=values[:,j]
    summary=dict(exposure=c['exposure'],point_in_time_snapshot=False,observation_availability_only=True,
        target='next_sessions_mean_squared_equal_weight_simple_basket_return',horizon=h,weights=[1/len(panel)]*len(panel),
        model_training_end=contract['train_end'],model_calibration_end=contract['calibration_end'],
        forecast_origins=len(frame),scored_origins=len(scored),pending_targets=int((~frame.scored).sum()),
        excluded_calibration_overlap=int((requested&~post_calibration).sum()),
        excluded_unavailable_origins=int((requested&post_calibration&~is_available).sum()),
        excluded_warmup_origins=int((requested&post_calibration&is_available&~warmed).sum()),
        common_return_period=[str(first.dates[i].date()) for i in (0,-1)],
        origin_period=[str(scored.date.iloc[i].date()) for i in (0,-1)],
        target_maturity_period=[str(scored.target_end.iloc[i].date()) for i in (0,-1)],
        volatility_boundaries=boundaries.tolist(),
        state_counts=np.bincount(scored.state,minlength=model.n_clusters).tolist(),
        volatility_state_counts=np.bincount(scored.volatility_state_label,minlength=3).tolist(),
        mapping_support={name:dict(minimum=int(scored[name].min()),median=float(scored[name].median()),maximum=int(scored[name].max()))
                         for name in ('regime_count','volatility_count')},
        forecast_floor_counts={method:int(result['floor_applied'][positions,j].sum()) for j,method in enumerate(METHODS)},
        metrics=_metrics(scored),by_origin_year={str(year):_metrics(group) for year,group in scored.groupby(scored.date.dt.year)},
        comparisons=paired_comparisons(losses,list(METHODS),blocks=c['bootstrap_blocks'],repeats=c['bootstrap_repeats'],seed=c['bootstrap_seed']),
        uncertainty='paired stationary-bootstrap percentile 95% pointwise intervals; conditional on frozen model and rule; no multiplicity adjustment')
    return frame,summary


def run_risk(bundle,config_path,output):
    contract,model=load_bundle(bundle)
    c=yaml.safe_load(Path(config_path).read_text())
    validate_risk_config(c,contract)
    panel={}
    for a in c['assets']:
        path=Path(a['dataset'])
        if digest(path)!=a['sha256']: raise ValueError('snapshot hash mismatch')
        check_acquisition(json.loads(path.with_suffix('.json').read_text()),a['symbol'],a['sha256'])
        value=load_csv(path,provider=c['provider'],price_column=c['price_column'])
        if value.dataset_sha256!=a['sha256']: raise ValueError('snapshot changed while loading')
        panel[a['symbol']]=value
    frame,summary=risk_panel(panel,c,contract,model)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    frame.to_parquet(output/'forecasts.parquet',index=False)
    write_json(output/'risk.json',dict(summary=summary,config=c,bundle_digest=digest(Path(bundle)/'checksums.json')))
    return dict(type='risk',**summary)
