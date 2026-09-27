# SPDX-License-Identifier: GPL-3.0-only
"""Causal second-moment forecasts and paired dependent loss comparisons."""
import numpy as np

from .robustness import stationary_indices

METHODS=('expanding','ewma','rolling','regime','volatility_state')


def _integer(value,name,minimum=1):
    if type(value) is not int or value<minimum:
        raise ValueError(f'invalid {name}')


def _states(values,n,k,window):
    x=np.asarray(values)
    if (x.shape!=(n,) or x.dtype.kind not in 'iu' or np.any(x[:window-1]!=-1)
            or np.any(x[window-1:]<0) or np.any(x[window-1:]>=k)):
        raise ValueError('invalid state history')
    return x


def causal_forecasts(q,states,volatility_states,*,k,window=63,horizon=5,min_history=252,
                     prior_count=252,ewma_decay=.94,forecast_floor=1e-12):
    """At t, state statistics contain only targets with maturity <= t.

    Historical labels may be retrospective fitted-model statistics. Callers must
    restrict assessment to origins after model selection and check availability.
    No realized future target is read to produce the current prediction.
    """
    for name,value in dict(k=k,window=window,horizon=horizon,min_history=min_history,prior_count=prior_count).items():
        _integer(value,name)
    if min_history<window: raise ValueError('minimum history below window')
    if type(ewma_decay) not in (int,float) or not 0<ewma_decay<1: raise ValueError('invalid EWMA decay')
    if type(forecast_floor) not in (int,float) or not 0<forecast_floor<np.inf: raise ValueError('invalid forecast floor')
    q=np.asarray(q,dtype=float)
    if q.ndim!=1 or len(q)<min_history or not np.isfinite(q).all() or np.any(q<0):
        raise ValueError('invalid squared-return history')
    n=len(q)
    state_arrays=(_states(states,n,k,window),_states(volatility_states,n,3,window))
    sums=[np.zeros(k),np.zeros(3)]
    counts=[np.zeros(k,dtype=int),np.zeros(3,dtype=int)]
    support=np.zeros((n,2),dtype=int)
    forecasts=np.full((n,len(METHODS)),np.nan)
    floor_applied=np.zeros_like(forecasts,dtype=bool)
    total=0.
    ewma=0.
    for t,value in enumerate(q):
        total+=value
        expanding=total/(t+1)
        origin=t-horizon
        if origin>=window-1:
            matured=float(np.mean(q[origin+1:t+1]))
            for j,labels in enumerate(state_arrays):
                state=labels[origin]
                sums[j][state]+=matured
                counts[j][state]+=1
        if t<min_history-1: continue
        ewma=expanding if t==min_history-1 else ewma_decay*ewma+(1-ewma_decay)*value
        predictions=[expanding,ewma,float(np.mean(q[t-window+1:t+1]))]
        for j,labels in enumerate(state_arrays):
            state=labels[t]
            support[t,j]=counts[j][state]
            predictions.append((sums[j][state]+prior_count*expanding)/(counts[j][state]+prior_count))
        predictions=np.asarray(predictions)
        if not np.isfinite(predictions).all(): raise ValueError('nonfinite risk forecast')
        floor_applied[t]=predictions<forecast_floor
        forecasts[t]=np.maximum(predictions,forecast_floor)
    return dict(methods=list(METHODS),forecasts=forecasts,regime_count=support[:,0],
                volatility_count=support[:,1],floor_applied=floor_applied)


def forecast_losses(target,forecasts):
    """QLIKE without target-only terms; valid at target=0. MSE is secondary."""
    y=np.asarray(target,dtype=float)
    f=np.asarray(forecasts,dtype=float)
    if (y.ndim!=1 or f.ndim!=2 or f.shape[0]!=len(y) or not len(y) or not f.shape[1]
            or not np.isfinite(y).all() or not np.isfinite(f).all() or np.any(y<0) or np.any(f<=0)):
        raise ValueError('invalid target or forecast')
    losses=dict(qlike=np.log(f)+y[:,None]/f,mse=(y[:,None]-f)**2)
    if any(not np.isfinite(v).all() for v in losses.values()): raise ValueError('nonfinite forecast loss')
    return losses


def paired_comparisons(losses,methods,*,blocks=(20,63,126),repeats=2000,seed=2026):
    """Pointwise percentile intervals, conditional on the fixed forecasting rule.

    One draw is shared across every method and loss; resample differences, not
    independent model rows. Circular stationary resampling assumes approximate
    stationarity, and does not refit or repeat model selection.
    """
    _integer(repeats,'bootstrap repeats',2)
    _integer(seed,'bootstrap seed',0)
    if not methods or len(set(methods))!=len(methods) or 'regime' not in methods or len(methods)<2 or not losses:
        raise ValueError('invalid comparison methods')
    arrays={name:np.asarray(value,dtype=float) for name,value in losses.items()}
    first=next(iter(arrays.values()))
    if first.ndim!=2: raise ValueError('invalid loss array')
    n=first.shape[0]
    if any(a.shape!=(n,len(methods)) or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('invalid loss array')
    if not blocks or len(set(blocks))!=len(blocks): raise ValueError('invalid block lengths')
    for block in blocks:
        _integer(block,'block length')
        if block>=n: raise ValueError('block length must be smaller than assessment')
    idx=methods.index('regime')
    columns=[(name,method) for name in arrays for method in methods if method!='regime']
    delta=np.column_stack([arrays[name][:,idx]-arrays[name][:,methods.index(method)] for name,method in columns])
    mean=delta.mean(axis=0)
    rows=[]
    for block in blocks:
        # Reset per block so reordering sensitivities cannot change a block's draws.
        rng=np.random.default_rng(np.random.SeedSequence([seed,block]))
        boot=np.empty((repeats,len(columns)))
        for i in range(repeats):
            indices,_=stationary_indices(n,block,rng)
            boot[i]=delta[indices].mean(axis=0)
        low,high=np.quantile(boot,[.025,.975],axis=0)
        rows.extend(dict(loss=name,baseline=method,mean_block=block,mean_difference=float(mean[j]),
                         lower=float(low[j]),upper=float(high[j]),replicates=repeats,confidence=.95)
                    for j,(name,method) in enumerate(columns))
    return rows
