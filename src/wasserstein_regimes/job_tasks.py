# SPDX-License-Identifier: GPL-3.0-only
"""Explicit local task types; no arbitrary commands or dynamically loaded callables."""
import json
from pathlib import Path
import time

import numpy as np
import yaml

from .artifact_store import write_json


def dispatch(spec,output):
    if spec['type']=='score':
        from .frozen import score_snapshot
        return dict(type='score',**score_snapshot(spec['bundle'],spec['config'],output))
    if spec['type']=='refit': return refit(spec,output)
    if spec['type']=='benchmark': return benchmark(spec)
    raise ValueError('unknown job type')


def refit(spec,output):
    from .experiments import code_provenance
    from .joint_market import load_panel,source_identity
    from .joint_validation import _reference,fit_returns
    from .robustness import partition_diagnostics
    c=yaml.safe_load(Path(spec['market_config']).read_text())
    panel,batch,masks,data=load_panel(c,c['validation_end'])
    reference,parent_digest=_reference(Path(spec['development']),c,source_identity(),code_provenance(),data)
    first=next(iter(panel.values()))
    mask=first.dates<=spec['train_end']
    vectors=np.column_stack([s.returns[mask] for s in panel.values()])
    fitted=fit_returns(vectors,c,reference.n_clusters,reference.projections_,seed=spec['seed'],candidates=spec['candidates'])
    anchors=batch.samples[masks['validation']]
    row=dict(type='refit',train_end=spec['train_end'],seed=spec['seed'],candidates=spec['candidates'],
        parent_digest=parent_digest,training_returns=len(vectors),scales=fitted.scales_.tolist(),
        converged=fitted.converged_,**partition_diagnostics(fitted.predict(anchors),reference.n_clusters,reference.predict(anchors)))
    write_json(output/'refit.json',row)
    return row


def benchmark(spec):
    from .sliced import SlicedWassersteinKMedoids
    rng=np.random.default_rng(spec['seed'])
    samples=rng.normal(size=(spec['n_windows'],spec['length'],spec['dimensions']))
    model=SlicedWassersteinKMedoids(n_clusters=spec['k'],n_projections=spec['projections'],
        candidate_size=spec['candidates'],n_init=spec['n_init'],random_state=spec['seed'])
    start=time.perf_counter()
    model.fit(samples)
    fitted=time.perf_counter()
    labels=model.predict(samples)
    scored=time.perf_counter()
    return dict(type='benchmark',input_bytes=samples.nbytes,fit_seconds=fitted-start,score_seconds=scored-fitted,
        counts=np.bincount(labels,minlength=spec['k']).tolist(),inertia=model.inertia_,converged=model.converged_,
        parameters=spec,scope='engineering benchmark on iid Gaussian windows, not regime recovery')
