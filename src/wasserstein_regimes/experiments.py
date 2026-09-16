# SPDX-License-Identifier: GPL-3.0-only
"""Chronological research orchestration; future outcomes never enter fitting."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from .clustering import WassersteinKMeans


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def artifact_id(config, provenance):
    return hashlib.sha256(canonical_json(dict(config=config, provenance=provenance)).encode()).hexdigest()[:20]


def split_masks(dates, price_start, *, train_end, validation_end, test_end, strict=True):
    dates, starts = pd.DatetimeIndex(dates), pd.DatetimeIndex(price_start)
    a, b, c = map(pd.Timestamp, (train_end, validation_end, test_end))
    if not a < b < c or len(dates) != len(starts):
        raise ValueError('Expected ordered boundaries and aligned temporal metadata')
    train = np.asarray(dates <= a)
    validation = np.asarray((dates > a) & (dates <= b))
    test = np.asarray((dates > b) & (dates <= c))
    if strict:
        # Strict raw-price intervals, including the first return's preceding price.
        if train.any():
            validation &= starts > dates[train].max()
        if np.any(dates <= b):
            test &= starts > dates[dates <= b].max()
    return dict(train=train, validation=validation, test=test)


def fit_primary(train, validation, *, k_candidates, seed=42, n_init=20):
    """Select K on validation silhouette in fixed full-quantile W2 geometry.

    This is descriptive separation selection, not evidence of predictive value.
    Validation does not update centroids. A smaller K wins an exact tie.
    """
    candidates = []
    for k in sorted(k_candidates):
        model = WassersteinKMeans(n_clusters=k, random_state=seed, n_init=n_init).fit(train)
        labels = model.predict(validation)
        score = None
        if 1 < len(np.unique(labels)) < len(labels):
            score = float(silhouette_score(np.sort(validation, axis=1), labels,
                                          sample_size=min(800, len(labels)), random_state=seed))
        candidates.append((model, dict(k=k, validation_silhouette=score, converged=model.converged_)))
    eligible = [pair for pair in candidates if pair[1]['validation_silhouette'] is not None]
    if not eligible:
        raise ValueError('No candidate has at least two occupied validation clusters')
    selected = max(eligible, key=lambda pair: pair[1]['validation_silhouette'])
    return selected[0], [row for _, row in candidates]


def code_provenance():
    """Hash tracked and untracked source inputs, excluding data/runtime artifacts."""
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args])
    try:
        sha = git('rev-parse', 'HEAD').decode().strip()
        names = git('ls-files', '--cached', '--others', '--exclude-standard', '-z').decode().split('\0')
        digest = hashlib.sha256()
        for name in sorted(set(filter(None, names))):
            path = root / name
            if path.is_file():
                digest.update(name.encode() + b'\0' + path.read_bytes() + b'\0')
        worktree_hash = digest.hexdigest()
    except (OSError, subprocess.CalledProcessError):
        sha, worktree_hash = 'unavailable', 'unavailable'
    packages = ['numpy', 'scipy', 'scikit-learn', 'pandas', 'pyarrow', 'hmmlearn', 'exchange-calendars']
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'unavailable'
    return dict(git_sha=sha, dirty_worktree_hash=worktree_hash, dependency_versions=versions)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(_json_safe(value), sort_keys=True, indent=2, allow_nan=False) + '\n')


def _fold_boundaries(year, validation_years, test_end):
    return dict(train_end=f'{year-validation_years-1}-12-31',
                validation_end=f'{year-1}-12-31', test_end=test_end)


def _summary(batch, labels, centers, distances, cal, returns, *, threshold, horizon, bandwidth, seed):
    from .evaluation import (component_fractions, future_outcomes, mmd2, novelty,
                             temporal_summary, moving_block_indices)
    from .transport import pairwise_distance, w2_decomposition
    k = len(centers)
    parts = w2_decomposition(batch.samples, centers)
    assigned = {name: matrix[np.arange(len(labels)), labels] for name, matrix in parts.items()}
    frame = pd.DataFrame(dict(date=batch.dates, available_at=batch.available_at,
                              endpoint=batch.endpoints, price_start=batch.price_start,
                              price_end=batch.price_end, regime=labels))
    for key, value in novelty(distances, cal, threshold=threshold).items():
        frame[key] = value
    for key, value in assigned.items():
        frame[f'w2_{key}'] = value
    for key, value in future_outcomes(returns, batch.endpoints, horizon=horizon).items():
        frame[key] = value
    upper = np.triu_indices(k, 1)
    between = pairwise_distance(centers, centers)[upper]
    centroid_parts = w2_decomposition(centers, centers)
    # Select chronologically separated windows, including their preceding prices.
    selected, last = [], -np.inf
    for i, endpoint in enumerate(batch.endpoints):
        if endpoint - last > batch.samples.shape[1]:
            selected.append(i)
            last = endpoint
    within_mmd, between_mmd = [], []
    for a, b in zip(selected[:-1], selected[1:]):
        value = mmd2(batch.samples[a], batch.samples[b], bandwidth=bandwidth)
        (within_mmd if labels[a] == labels[b] else between_mmd).append(value)
    occupancy = np.bincount(labels, minlength=k) / len(labels)
    conditional = []
    rng = np.random.default_rng(seed)
    block = min(len(frame), batch.samples.shape[1] + horizon)
    draws = [moving_block_indices(len(frame), block_length=block, rng=rng) for _ in range(100)]
    for state in range(k):
        mask = labels == state
        row = dict(regime=state, n=int(mask.sum()), occupancy=float(occupancy[state]),
                   decomposition=component_fractions({name: values[mask] for name, values in assigned.items()}))
        for key in ('forward_log_return', 'forward_volatility', 'forward_drawdown', 'forward_downside'):
            values = frame[key].to_numpy()
            valid = mask & np.isfinite(values)
            estimates = []
            for indices in draws:
                good = (labels[indices] == state) & np.isfinite(values[indices])
                if good.any():
                    estimates.append(float(values[indices][good].mean()))
            row[key] = dict(mean=float(values[valid].mean()) if valid.any() else None,
                            n=int(valid.sum()), block_interval=np.quantile(estimates, [.025,.975]).tolist() if estimates else None)
        conditional.append(row)
    metrics = dict(n=len(labels), occupancy=occupancy.tolist(), within_w2=float(np.sqrt(assigned['total']).mean()),
                   between_centroid_w2_mean=float(between.mean()) if len(between) else None,
                   centroid_decomposition=component_fractions({key: value[upper] for key,value in centroid_parts.items()}),
                   assignment_decomposition=component_fractions(assigned), temporal=temporal_summary(labels, batch.endpoints, k=k),
                   mmd=dict(bandwidth=bandwidth, estimator='biased Gaussian squared MMD; consecutive separated windows',
                            within_mean=float(np.mean(within_mmd)) if within_mmd else None,
                            between_mean=float(np.mean(between_mmd)) if between_mmd else None,
                            within_pairs=len(within_mmd), between_pairs=len(between_mmd)),
                   ood_rate=float(frame.ood.mean()), conditional_outcomes=conditional)
    return frame, metrics, parts


def run_fold(series, batch, config, *, year, test_end, output):
    from scipy.stats import norm
    from sklearn.metrics import adjusted_rand_score
    from .baselines import FeatureBaseline, CausalGaussianHMM
    from .transport import pairwise_distance, standardize_windows
    bounds = _fold_boundaries(year, config['validation_years'], test_end)
    strict = split_masks(batch.dates, batch.price_start, **bounds, strict=True)
    operational = split_masks(batch.dates, batch.price_start, **bounds, strict=False)
    train_indices = np.flatnonzero(strict['train'])[::config['fit_stride']]
    train = batch.samples[train_indices]
    validation = batch.samples[strict['validation']]
    if not len(train) or len(validation) < 2 or not strict['test'].any():
        raise ValueError(f'Insufficient data for fold {year}')
    primary, selection = fit_primary(train, validation, k_candidates=config['k_candidates'],
                                     seed=config['seed'], n_init=config['n_init'])
    k = primary.n_clusters
    output.mkdir(parents=True, exist_ok=True)
    primary.save(output / 'model')
    np.savez_compressed(output / 'centroids.npz', centers=primary.centers_)
    train_return_mask = series.dates <= pd.Timestamp(bounds['train_end'])
    bandwidth = max(float(np.nanstd(series.returns[train_return_mask])), 1e-10)
    models, summaries, frames, labels_by_model = {}, {}, [], {}
    for name in config['models']:
        print(f'  fold {year}: {name}', flush=True)
        transform = standardize_windows if name == 'shape_w2' else np.asarray
        tr, val = transform(train), transform(validation)
        if name == 'w2':
            model = primary
        elif name in ('w1', 'shape_w2'):
            model = WassersteinKMeans(metric='w1' if name == 'w1' else 'w2', n_clusters=k,
                                      random_state=config['seed'], n_init=config['n_init']).fit(tr)
        elif name != 'hmm':
            model = FeatureBaseline(kind=name, n_clusters=k, random_state=config['seed'], n_init=config['n_init']).fit(tr)
        else:
            observed = series.returns[train_return_mask]
            if not np.isfinite(observed).all():
                raise ValueError('HMM study requires contiguous training returns; segment-aware fitting is not implemented')
            model = CausalGaussianHMM(n_clusters=k, random_state=config['seed'], n_init=5).fit(observed)
            # Filter forward from the fitted training terminal state. No smoothing.
            remaining = series.returns[~train_return_mask]
            if not np.isfinite(remaining).all():
                raise ValueError('HMM study requires contiguous replay returns')
            probabilities = np.vstack([model.filter_proba(observed), model.filter_proba(remaining, initial=model.terminal_proba_)])
            hmm_labels = probabilities.argmax(axis=1)
            quantiles = norm.ppf((np.arange(config['window_length'])+.5)/config['window_length'])
            centers = (model.train_mean_ + model.train_std_ *
                       (model.means_[:,None] + np.sqrt(model.covars_)[:,None] * quantiles))
            model.centers_ = centers
            write_json(output / 'hmm.json', model.to_dict())
        models[name] = model
        if name not in ('w2','w1','shape_w2','hmm'):
            fitted=model.model_
            metadata=dict(kind=name,scaler_mean=model.scaler_.mean_,scaler_scale=model.scaler_.scale_,
                          label_map=model._label_map_,n_iter=getattr(fitted,'n_iter_',None),
                          converged=getattr(fitted,'converged_',None),random_state=config['seed'])
            for key in ('cluster_centers_','weights_','means_','covariances_','precisions_cholesky_'):
                if hasattr(fitted,key):
                    metadata[key]=getattr(fitted,key)
            write_json(output / f'{name}_model.json',metadata)
        centers = model.centers_
        # Feature and HMM novelty describe distributional distance to their prototypes,
        # not their own assignment probability. W1 uses W1; shape W2 uses standardized units.
        metric = 'w1' if name == 'w1' else 'w2'
        cal = pairwise_distance(val, centers, metric=metric).min(axis=1)
        if name in ('w1', 'shape_w2'):
            model.save(output / name)
        np.savez_compressed(output / f'{name}_prototypes.npz', centers=centers)
        summaries[name] = dict(converged=getattr(model, 'converged_', None),
                               effective_clusters=len(centers), distance_geometry=metric,
                               shape_standardized=name == 'shape_w2')
        for policy, masks in (('strict',strict), ('operational', operational)):
            scored = batch.subset(masks['test'])
            x = transform(scored.samples)
            labels = hmm_labels[scored.endpoints] if name == 'hmm' else model.predict(x)
            distances = pairwise_distance(x, centers, metric=metric)
            # Decomposition for shape model uses its own standardized geometry.
            from dataclasses import replace
            geometry_batch = replace(scored, samples=x)
            frame, metrics, parts = _summary(geometry_batch, labels, centers, distances, cal, series.returns,
                                            threshold=config['novelty_threshold'], horizon=config['horizon'],
                                            bandwidth=1. if name == 'shape_w2' else bandwidth, seed=config['seed'])
            frame['fold'], frame['model'], frame['policy'] = year, name, policy
            frames.append(frame)
            summaries[name][policy] = metrics
            if policy == 'strict':
                labels_by_model[name] = labels
                np.savez_compressed(output / f'{name}_decomposition.npz', **parts)
    comparison = {name: float(adjusted_rand_score(labels_by_model['w2'], labels)) for name,labels in labels_by_model.items()}
    return dict(year=year, boundaries=bounds, k=k, selection=selection, models=summaries,
                ari_against_raw_w2=comparison, training_windows=len(train), validation_windows=len(validation)), pd.concat(frames, ignore_index=True), models


def stability_study(series, batch, config, *, year, test_end):
    """Training-return block resampling with validation-only reference scoring."""
    from sklearn.metrics import adjusted_rand_score
    from .baselines import FeatureBaseline
    from .evaluation import moving_block_indices, align_centroids
    from .transport import standardize_windows, pairwise_distance
    bounds = _fold_boundaries(year, config['validation_years'], test_end)
    masks = split_masks(batch.dates, batch.price_start, **bounds)
    train = batch.samples[masks['train']][::config['fit_stride']]
    validation = batch.samples[masks['validation']]
    primary, _ = fit_primary(train, validation, k_candidates=config['k_candidates'], seed=config['seed'], n_init=config['n_init'])
    k = primary.n_clusters
    raw = series.returns[series.dates <= pd.Timestamp(bounds['train_end'])]
    if not np.isfinite(raw).all():
        raise ValueError('Block study currently requires contiguous training returns')
    output = dict(scoring_period='validation only', block_length=config['bootstrap_block_length'], models={})
    for name in ('w2','shape_w2','volatility','moments'):
        print(f'  stability: {name}', flush=True)
        transform = standardize_windows if name == 'shape_w2' else np.asarray
        def fit(samples, seed):
            if name in ('w2','shape_w2'):
                return WassersteinKMeans(n_clusters=k, n_init=config['n_init'],random_state=seed).fit(transform(samples))
            return FeatureBaseline(kind=name,n_clusters=k,n_init=config['n_init'],random_state=seed).fit(samples)
        reference = fit(train, config['seed'])
        reference_k = len(reference.centers_)
        ref_labels = reference.predict(transform(validation))
        groups = {}
        for group, repeats in (('seed',config['seed_repeats']),('block_bootstrap',config['bootstrap_repeats'])):
            rows=[]
            for repeat in range(repeats):
                seed=config['seed']+1000+repeat
                samples=train
                if group=='block_bootstrap':
                    indices=moving_block_indices(len(raw),block_length=min(len(raw),config['bootstrap_block_length']),rng=np.random.default_rng(seed))
                    samples=np.lib.stride_tricks.sliding_window_view(raw[indices],config['window_length'])[::config['fit_stride']]
                model=fit(samples,seed)
                effective_k = len(model.centers_)
                labels=model.predict(transform(validation))
                if effective_k == reference_k:
                    mapping=align_centroids(reference.centers_,model.centers_)
                    displacement=float(pairwise_distance(model.centers_,reference.centers_)[np.arange(reference_k),mapping].mean())
                    occupancy=np.bincount(mapping[labels],minlength=reference_k)/len(labels)
                    occupancy_l1=float(np.abs(occupancy-np.bincount(ref_labels,minlength=reference_k)/len(ref_labels)).sum())
                else:
                    displacement,occupancy_l1=None,None
                rows.append(dict(ari=float(adjusted_rand_score(ref_labels,labels)),effective_clusters=effective_k,
                                 centroid_displacement=displacement,occupancy_l1=occupancy_l1))
            groups[group]=dict(repetitions=repeats,reference_effective_clusters=reference_k,
                               ari_mean=float(np.mean([r['ari'] for r in rows])),
                               ari_range=np.quantile([r['ari'] for r in rows],[0,1]).tolist(),draws=rows)
        output['models'][name]=groups
    return output


def overlap_study(series, batch, config, *, year, test_end):
    from .evaluation import temporal_summary
    from sklearn.metrics import adjusted_rand_score
    bounds=_fold_boundaries(year,config['validation_years'],test_end)
    masks=split_masks(batch.dates,batch.price_start,**bounds)
    train=batch.samples[masks['train']]
    validation=batch.samples[masks['validation']]
    primary,_=fit_primary(train[::config['fit_stride']],validation,k_candidates=config['k_candidates'],seed=config['seed'],n_init=config['n_init'])
    raw=series.returns[series.dates<=pd.Timestamp(bounds['validation_end'])]
    if not np.isfinite(raw).all():
        raise ValueError('Overlap null requires contiguous development returns')
    n_train=int(np.sum(series.dates<=pd.Timestamp(bounds['train_end'])))
    length=config['window_length']
    rows=[]
    for stride in config['fit_strides']:
        print(f'  overlap stride {stride}',flush=True)
        model=WassersteinKMeans(n_clusters=primary.n_clusters,n_init=config['n_init'],random_state=config['seed']).fit(train[::stride])
        observed=temporal_summary(model.predict(validation))
        null=[]
        for repeat in range(config['null_repeats']):
            rng=np.random.default_rng(config['seed']+repeat+2000)
            # IID bootstrap from pooled development marginal removes serial dependence.
            stationary=rng.choice(raw,len(raw),replace=True)
            windows=np.lib.stride_tricks.sliding_window_view(stationary,length)
            tr=windows[:n_train-length+1:stride]
            val=windows[n_train+1:]
            nm=WassersteinKMeans(n_clusters=primary.n_clusters,n_init=config['n_init'],random_state=config['seed']+repeat).fit(tr)
            null.append(temporal_summary(nm.predict(val))['mean_dwell'])
        rows.append(dict(fit_stride=stride,score_stride=1,observed_mean_dwell=observed['mean_dwell'],
                         null_mean_dwell=float(np.mean(null)),null_interval=np.quantile(null,[.025,.975]).tolist(),
                         excess_mean_dwell=observed['mean_dwell']-float(np.mean(null)),
                         ari_vs_primary=float(adjusted_rand_score(primary.predict(validation),model.predict(validation))),
                         null_draws=null))
    return dict(period='validation only',null='IID resampling of development marginal; overlapping windows retained',
                n_init_observed=config['n_init'],n_init_null=config['n_init'],repetitions=config['null_repeats'],rows=rows)


def run(config_path, *, output_root='artifacts', stage='development'):
    """Run immutable local-data research; holdout access must be explicit."""
    import yaml
    from threadpoolctl import threadpool_limits
    from .data import load_csv
    from .windows import make_windows
    config=yaml.safe_load(Path(config_path).read_text())
    if stage not in ('development','holdout'):
        raise ValueError('stage must be development or holdout')
    config=dict(config,stage=stage)
    source=Path(config['dataset'])
    actual=hashlib.sha256(source.read_bytes()).hexdigest()
    if actual!=config['dataset_sha256']:
        raise ValueError('Dataset hash differs from the frozen specification')
    series=load_csv(source,price_column=config['price_column'],provider=config['provider'])
    # Drop future observations from the actual input to all development calculations.
    cutoff=pd.Timestamp(config['holdout_start'])-pd.Timedelta(days=1) if stage=='development' else pd.Timestamp(config['data_cutoff'])
    from dataclasses import replace
    mask=series.dates<=cutoff
    series=replace(series,returns=series.returns[mask],dates=series.dates[mask],available_at=series.available_at[mask],
                   price_start=series.price_start[mask],price_end=series.price_end[mask])
    batch=make_windows(series,length=config['window_length'],stride=config['score_stride'])
    provenance=code_provenance()
    provenance.update(dataset_sha256=actual,provider=series.provider,return_basis=series.return_basis,
                      data_cutoff=str(cutoff.date()),quality=dict(series.quality))
    run_id=artifact_id(config,provenance)
    output=Path(output_root)/run_id
    if (output/'checksums.json').exists():
        verify_artifact(output)
        print(f'Reusing completed immutable run {run_id}',flush=True)
        return output
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'config.json',config)
    write_json(output/'manifest.json',dict(provenance,run_id=run_id,window_length=config['window_length'],
                                         fit_stride=config['fit_stride'],score_stride=config['score_stride'],
                                         metric=config['metric'],seed=config['seed'],point_in_time=False))
    years=config['development_years'] if stage=='development' else [pd.Timestamp(config['holdout_start']).year]
    folds,frames=[],[]
    with threadpool_limits(limits=1):
        previous=None
        for year in years:
            end=f'{year}-12-31' if stage=='development' else config['data_cutoff']
            summary,frame,models=run_fold(series,batch,config,year=year,test_end=end,output=output/'models'/str(year))
            if previous is not None:
                from .evaluation import align_centroids
                from .transport import pairwise_distance
                from sklearn.metrics import adjusted_rand_score
                ref=previous
                current=models['w2']
                if len(ref.centers_)==len(current.centers_):
                    mapping=align_centroids(ref.centers_,current.centers_)
                    common=batch.samples[(batch.dates>pd.Timestamp(summary['boundaries']['train_end'])) & (batch.dates<=pd.Timestamp(summary['boundaries']['validation_end']))]
                    summary['refit']=dict(mapping=mapping.tolist(),ari=float(adjusted_rand_score(ref.predict(common),current.predict(common))),
                                          centroid_displacement=float(pairwise_distance(current.centers_,ref.centers_)[np.arange(len(mapping)),mapping].mean()),
                                          comparison='common current validation windows; descriptive, not independent test')
                else:
                    summary['refit']=dict(reason='Selected K changed; no bijective alignment')
            previous=models['w2']
            folds.append(summary);frames.append(frame)
        latest=years[-1]
        sensitivity=dict(stability=stability_study(series,batch,config,year=latest,test_end=str(cutoff.date())),
                         overlap=overlap_study(series,batch,config,year=latest,test_end=str(cutoff.date())))
    import shutil
    shutil.copyfile(output/'models'/str(years[-1])/'model.json',output/'model.json')
    shutil.copyfile(output/'models'/str(years[-1])/'centroids.npz',output/'centroids.npz')
    pd.concat(frames,ignore_index=True).to_parquet(output/'assignments.parquet',index=False)
    write_json(output/'metrics.json',dict(folds=folds,**sensitivity))
    manifest=json.loads((output/'manifest.json').read_text())
    manifest.update(k=[f['k'] for f in folds],split_boundaries=[f['boundaries'] for f in folds],
                    training_period=[str(series.dates[0].date()),folds[-1]['boundaries']['train_end']],
                    validation_period=[folds[-1]['boundaries']['train_end'],folds[-1]['boundaries']['validation_end']],
                    test_period=[folds[-1]['boundaries']['validation_end'],folds[-1]['boundaries']['test_end']])
    write_json(output/'manifest.json',manifest)
    seal_artifact(output)
    return output


def seal_artifact(path):
    """Record immutable research-file hashes; derived reports are excluded."""
    path=Path(path)
    files={str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(path.rglob('*')) if p.is_file() and p.suffix in ('.json','.npz','.parquet') and p.name!='checksums.json'}
    write_json(path/'checksums.json',files)


def verify_artifact(path):
    path=Path(path)
    try:
        files=json.loads((path/'checksums.json').read_text())
        for name,expected in files.items():
            target=(path/name).resolve()
            if not target.is_relative_to(path.resolve()) or hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
                raise ValueError('Artifact integrity check failed')
    except (OSError,json.JSONDecodeError) as exc:
        raise ValueError('Artifact integrity check failed') from exc
