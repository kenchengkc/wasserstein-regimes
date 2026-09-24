# SPDX-License-Identifier: GPL-3.0-only
"""Validation-only robustness runner bound to a sealed exploratory market study."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from .experiments import artifact_id, code_provenance, _json_safe, seal_artifact, verify_artifact, write_json
from .joint_controls import run_controls
from .joint_market import load_panel, source_identity, validate_config as validate_market_config
from .robustness import contaminate, partition_diagnostics, return_windows, stationary_indices
from .sliced import SlicedWassersteinKMedoids
from .transport import _positive_int

_FIELDS = {'schema_version', 'market_config', 'block_lengths', 'bootstrap_repeats', 'chronological_ends',
           'candidate_sizes', 'optimizer_seeds', 'synthetic_seeds', 'outlier_fractions', 'outlier_magnitude', 'seed'}


def validate_config(c):
    if not isinstance(c, dict) or set(c) != _FIELDS or type(c['schema_version']) is not int or c['schema_version'] != 1:
        raise ValueError('invalid validation schema or unknown fields')
    if not isinstance(c['market_config'], str) or not c['market_config']:
        raise ValueError('market configuration path required')
    for name in ('bootstrap_repeats', 'seed'):
        _positive_int(c[name], name)
    for name in ('block_lengths', 'candidate_sizes', 'optimizer_seeds', 'synthetic_seeds'):
        if not isinstance(c[name], list) or not c[name]:
            raise ValueError(f'invalid {name}')
        for value in c[name]:
            _positive_int(value, name)
        if len(set(c[name])) != len(c[name]):
            raise ValueError(f'duplicate {name}')
    if min(c['candidate_sizes']) < 2:
        raise ValueError('candidate budget below two')
    if (not isinstance(c['outlier_fractions'], list) or not c['outlier_fractions']
            or any(type(v) not in (int, float) or not np.isfinite(v) or not 0 < v <= 1 for v in c['outlier_fractions'])
            or len(set(c['outlier_fractions'])) != len(c['outlier_fractions'])):
        raise ValueError('invalid outlier fractions')
    value = c['outlier_magnitude']
    if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
        raise ValueError('invalid outlier magnitude')
    dates = c['chronological_ends']
    if not isinstance(dates, list) or not dates or any(not isinstance(d, str) for d in dates):
        raise ValueError('chronological ends must be date strings')
    parsed = pd.DatetimeIndex(dates)
    if parsed.hasnans or parsed.tz is not None or not parsed.is_monotonic_increasing or parsed.has_duplicates:
        raise ValueError('invalid chronological ends')


def fit_returns(vectors, c, k, directions, *, seed=None, candidates=None, scales=None):
    scale = np.std(vectors, axis=0) if scales is None else scales
    model = SlicedWassersteinKMedoids(n_clusters=k, candidate_size=c['candidate_size'] if candidates is None else candidates,
        n_init=c['n_init'], chunk_size=c['chunk_size'], random_state=c['seed'] if seed is None else seed,
        scales=scale, projections=directions)
    return model.fit(return_windows(vectors, c['window_length'], c['fit_stride']))


def market_diagnostics(vectors, dates, anchors, reference, mc, c):
    labels = reference.predict(anchors)
    rows = []
    k, directions = reference.n_clusters, reference.projections_
    def record(kind, model, **metadata):
        rows.append(dict(kind=kind, **metadata, **partition_diagnostics(model.predict(anchors), k, labels),
                         converged=model.converged_, candidate_converged=[r.get('converged', False) for r in model.candidate_runs_],
                         training_windows=len(model.labels_), scales=model.scales_.tolist(),
                         mean_squared_nearest_distance=float(np.mean(model.transform(anchors).min(axis=1)**2))))
    for length in c['block_lengths']:
        for repeat in range(c['bootstrap_repeats']):
            print(f'return bootstrap length={length} repeat={repeat}', flush=True)
            rng = np.random.default_rng(np.random.SeedSequence([c['seed'], length, repeat]))
            indices, seams = stationary_indices(len(vectors), length, rng)
            model = fit_returns(vectors[indices], mc, k, directions)
            # The first row of a window is not an internal boundary.
            crossing = np.lib.stride_tricks.sliding_window_view(seams, mc['window_length'])[:, 1:].any(axis=1)[::mc['fit_stride']]
            record('return_bootstrap', model, mean_block=length, repeat=repeat,
                   seams=int(seams[1:].sum()), windows_crossing_seams=int(crossing.sum()))
    for end in c['chronological_ends']:
        selected = dates <= end
        print(f'chronological refit through {end}', flush=True)
        record('chronological', fit_returns(vectors[selected], mc, k, directions), train_end=end,
               training_returns=int(selected.sum()), is_reference=end == mc['train_end'])
    for budget in c['candidate_sizes']:
        for seed in c['optimizer_seeds']:
            print(f'candidate budget={budget} seed={seed}', flush=True)
            model = fit_returns(vectors, mc, k, directions, seed=seed, candidates=budget, scales=reference.scales_)
            record('candidate_budget', model, candidates=budget, seed=seed,
                   is_reference=budget == mc['candidate_size'] and seed == mc['seed'])
    for fraction in c['outlier_fractions']:
        contaminated, count = contaminate(vectors, fraction, c['outlier_magnitude'], np.random.default_rng(c['seed']))
        record('training_outliers', fit_returns(contaminated, mc, k, directions), fraction=fraction,
               contaminated_returns=count, magnitude=c['outlier_magnitude'])
    return rows


def _verified(path, required):
    verify_artifact(path)
    inventory = json.loads((path/'checksums.json').read_text())
    if not required <= set(inventory):
        raise ValueError('incomplete artifact checksum inventory')


def _reference(dev, mc, sources, provenance, data):
    _verified(dev, {'config.json', 'manifest.json', 'metrics.json', 'models/scaled_joint.npz'})
    manifest = json.loads((dev/'manifest.json').read_text())
    if json.loads((dev/'config.json').read_text()) != mc or manifest['stage'] != 'development':
        raise ValueError('parent development protocol differs')
    prior_sources = manifest['source_files']
    if not {'sliced.py', 'joint_market.py', 'data.py', 'transport.py'} <= set(prior_sources):
        raise ValueError('parent numerical source inventory missing')
    if any(sources.get(name) != digest for name, digest in prior_sources.items() if name != 'cli.py'):
        raise ValueError('parent numerical source differs')
    if manifest['dependency_versions'] != provenance['dependency_versions']:
        raise ValueError('parent dependencies differ')
    if manifest['data_identity'] != artifact_id(_json_safe(data), {}):
        raise ValueError('parent acquisition identity differs')
    model = SlicedWassersteinKMedoids.load(dev/'models/scaled_joint.npz')
    if model.n_clusters != json.loads((dev/'metrics.json').read_text())['fit']['k']:
        raise ValueError('parent model K differs')
    return model, hashlib.sha256((dev/'checksums.json').read_bytes()).hexdigest()


def _render(target):
    metrics = json.loads((target/'metrics.json').read_text())
    table = [{key:row.get(key) for key in ('kind', 'mean_block', 'repeat', 'train_end', 'candidates', 'seed', 'fraction', 'ari', 'occupied')}
             for row in metrics['market']]
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Joint regime robustness</title>'
    page += '<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px}td,th{padding:6px}pre{white-space:pre-wrap}</style>'
    page += '<h1>Joint regime robustness</h1><p>Exploratory validation only. Forced clusters are not proof of regimes; ARI is agreement, not accuracy. Resampling conditions on frozen K and directions. No confidence interval or predictive claim.</p>'
    page += pd.DataFrame(table).to_html(index=False, escape=True)
    page += '<details><summary>Every synthetic and market result</summary><pre>'+html.escape(json.dumps(metrics, indent=2))+'</pre></details></html>'
    (target/'report.html').write_text(page)
    return target/'report.html'


def run_joint_validation(config_path, *, development, output_root='artifacts'):
    c = yaml.safe_load(Path(config_path).read_text())
    validate_config(c)
    mc = yaml.safe_load(Path(c['market_config']).read_text())
    validate_market_config(mc)
    if any(pd.Timestamp(end) > pd.Timestamp(mc['train_end']) for end in c['chronological_ends']):
        raise ValueError('chronological refit cannot extend beyond training')
    dev = Path(development)
    provenance, sources = code_provenance(), source_identity()
    panel, batch, masks, data = load_panel(mc, mc['validation_end'])
    reference, digest = _reference(dev, mc, sources, provenance, data)
    if min(c['candidate_sizes']) < reference.n_clusters:
        raise ValueError('candidate budget smaller than frozen K')
    identity = dict(parent_digest=digest, source_files=sources, dependencies=provenance['dependency_versions'])
    combined = dict(validation=c, market=mc)
    target = Path(output_root)/f'joint-validation-{artifact_id(combined, identity)}'
    if target.exists():
        _verified(target, {'config.json', 'manifest.json', 'metrics.json'})
        return _render(target)
    first = next(iter(panel.values()))
    train_mask = first.dates <= mc['train_end']
    vectors = np.column_stack([v.returns[train_mask] for v in panel.values()])
    anchors = batch.samples[masks['validation']]
    if len(anchors) < 2:
        raise ValueError('insufficient validation anchors')
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=target.name+'-partial-', dir=target.parent))
    with threadpool_limits(limits=1):
        market = market_diagnostics(vectors, first.dates[train_mask], anchors, reference, mc, c)
        synthetic = run_controls(c)
    write_json(temp/'config.json', combined)
    write_json(temp/'manifest.json', dict(**identity, execution_provenance=provenance, data=data,
        exposure='exploratory_validation_only', training_returns=len(vectors), reference_k=reference.n_clusters,
        anchor_period=[str(batch.dates[masks['validation']][i].date()) for i in (0, -1)],
        anchor_windows=len(anchors), asset_order=list(panel),
        resampling_scope='return vectors and scales refitted; K and projections fixed; not confidence intervals'))
    write_json(temp/'metrics.json', dict(market=market, synthetic=synthetic))
    seal_artifact(temp)
    _render(temp)
    temp.rename(target)
    return target/'report.html'
