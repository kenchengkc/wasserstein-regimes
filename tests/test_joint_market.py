# SPDX-License-Identifier: GPL-3.0-only
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from wasserstein_regimes.data import ReturnSeries
from wasserstein_regimes.joint_market import validate_config, prepare_panel, training_scales, run_joint_market


def config():
    return yaml.safe_load(Path('configs/joint_market.yaml').read_text())


def series(start='2018-01-01', end='2025-01-01'):
    prices=pd.bdate_range(start,end)
    n=len(prices)-1
    rng=np.random.default_rng(2)
    return ReturnSeries(rng.normal(size=n),prices[1:],
                        prices[1:].tz_localize('UTC')+pd.Timedelta(hours=21),
                        prices[:-1],prices[1:])


def test_schema_rejects_unknown_keys_and_wrong_exposure():
    c=config()
    validate_config(c)
    with pytest.raises(ValueError): validate_config(dict(c,unknown=3))
    with pytest.raises(ValueError): validate_config(dict(c,exposure='untouched_holdout'))
    with pytest.raises(ValueError): validate_config(dict(c,k_candidates=[True,2]))


def test_common_history_scaling_and_purge_ignore_future():
    c=config()
    a=series()
    b=series('2019-01-01')
    panel,batch,masks=prepare_panel({'A':a,'B':b},c,'2024-12-31')
    assert batch.symbols==('A','B')
    assert panel['A'].dates.equals(panel['B'].dates)
    assert batch.price_start[masks['validation']].min()>batch.dates[masks['train']].max()
    assert batch.price_start[masks['test']].min()>batch.dates[batch.dates<=c['validation_end']].max()
    changed=a.returns.copy()
    changed[a.dates>c['train_end']]*=100
    other,_,_=prepare_panel({'A':replace(a,returns=changed),'B':b},c,'2024-12-31')
    np.testing.assert_allclose(training_scales(panel,c['train_end']),training_scales(other,c['train_end']))


def test_assessment_requires_development_before_any_execution(tmp_path):
    with pytest.raises(ValueError,match='development'):
        run_joint_market('configs/joint_market.yaml',stage='assessment',output_root=tmp_path)


def test_panel_rejects_calendar_mismatch():
    a=series()
    b=replace(a,dates=a.dates.delete(10),returns=np.delete(a.returns,10),
              available_at=a.available_at.delete(10),price_start=a.price_start.delete(10),price_end=a.price_end.delete(10))
    with pytest.raises(ValueError,match='sessions'):
        prepare_panel({'A':a,'B':b},config(),'2023-12-31')


def test_hash_mismatch_fails_before_loading_market_data(tmp_path):
    from wasserstein_regimes.joint_market import load_panel
    c=config()
    source=tmp_path/'changed.csv'
    source.write_text('changed')
    c['assets'][0]['dataset']=str(source)
    with pytest.raises(ValueError,match='hash mismatch'):
        load_panel(c,c['validation_end'])


def test_assessment_only_reloads_frozen_models(tmp_path,monkeypatch):
    import json
    import wasserstein_regimes.joint_market as module
    from wasserstein_regimes.experiments import verify_artifact
    c=config()
    c.update(window_length=5,k_candidates=[2],fit_stride=5,n_projections=8,candidate_size=16,
             n_init=1,selection_sample=40,baseline_n_init=2,hmm_n_init=1,hmm_max_iter=20,
             projection_counts=[8],projection_seeds=[42],bootstrap_repeats=1,bootstrap_block_windows=3)
    c['assets']=c['assets'][:2]
    base=series()
    rng=np.random.default_rng(9)
    scale=np.where(np.arange(len(base.returns))//60%2,3.,.5)
    all_series={symbol:replace(base,returns=rng.normal(size=len(scale))*scale)
                for symbol in ['SPY','QQQ']}
    source_meta={'synthetic':True}
    def fake_load(config,cutoff):
        panel,batch,masks=prepare_panel(all_series,config,cutoff)
        return panel,batch,masks,source_meta
    monkeypatch.setattr(module,'load_panel',fake_load)
    cfg=tmp_path/'config.yaml'
    cfg.write_text(yaml.safe_dump(c))
    dev=run_joint_market(cfg,output_root=tmp_path).parent
    verify_artifact(dev)
    development_metrics=json.loads((dev/'metrics.json').read_text())
    for row in development_metrics['sensitivity']['projections']:
        assert sum(row['counts'])==sum(row['reference_counts'])
        assert sum(row['occupancy'])==pytest.approx(1.)
    monkeypatch.setattr(module,'_fit_models',lambda *a,**k:pytest.fail('assessment refitted models'))
    assessment=run_joint_market(cfg,stage='assessment',development=dev).parent
    verify_artifact(assessment)
    data=json.loads((assessment/'metrics.json').read_text())
    assert data['stage']=='assessment'
    assert len({m['n'] for m in data['models'].values()})==1
    assert data['sensitivity'] is None
    source_meta['revision']=2
    with pytest.raises(ValueError,match='acquisition'):
        run_joint_market(cfg,stage='assessment',development=dev)
    del source_meta['revision']
    from wasserstein_regimes.experiments import seal_artifact
    (dev/'audit.json').write_text('different sealed development evidence')
    seal_artifact(dev)
    with pytest.raises(ValueError,match='development.*digest'):
        run_joint_market(cfg,stage='assessment',development=dev)
    c['fit_stride']=6
    cfg.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError,match='protocol'):
        run_joint_market(cfg,stage='assessment',development=dev)


def test_legacy_spy_acquisition_hash_and_conflicts():
    from wasserstein_regimes.joint_market import check_acquisition
    check_acquisition({'symbol':'SPY','dataset_sha256':'abc'},'SPY','abc')
    check_acquisition({'symbol':'SPY','sha256':'abc'},'SPY','abc')
    with pytest.raises(ValueError):
        check_acquisition({'symbol':'SPY','sha256':'abc','dataset_sha256':'different'},'SPY','abc')
