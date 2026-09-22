import importlib
import importlib.util

import numpy as np
import pytest
from wasserstein_regimes import WassersteinKMeans


def module():
    assert importlib.util.find_spec('wasserstein_regimes.cross_asset') is not None
    return importlib.import_module('wasserstein_regimes.cross_asset')


def test_protocol_rejects_asset_specific_tuning():
    m=module()
    common={'window_length':63,'seed':42}
    m.validate_protocol({'A':dict(common,symbol='A',dataset='a',dataset_sha256='aa'),
                         'B':dict(common,symbol='B',dataset='b',dataset_sha256='bb')})
    with pytest.raises(ValueError,match='window_length'):
        m.validate_protocol({'A':common,'B':dict(common,window_length=21)})


def test_refit_comparison_respects_permutation_and_standardization():
    m=module()
    samples=np.array([[-2,0,1,1],[-1,-1,0,2],[-2,0,1,1],[-1,-1,0,2]],float)
    from wasserstein_regimes import standardize_windows
    x=standardize_windows(samples)
    a=WassersteinKMeans(n_clusters=2,n_init=1,initialization=x[:2]).fit(x)
    b=WassersteinKMeans(n_clusters=2,n_init=1,initialization=x[:2][::-1]).fit(x)
    row=m.compare_refits(a,b,10+3*samples,shape_only=True)
    assert row['ari']==1
    assert row['centroid_displacement']==pytest.approx(0.,abs=1e-14)
    assert not row['both_single_state']


def test_refits_with_changed_k_keep_ari_but_no_bijection():
    m=module()
    x=np.array([[0,0],[1,1],[4,4],[5,5],[10,10],[11,11]],float)
    a=WassersteinKMeans(n_clusters=2,n_init=2).fit(x)
    b=WassersteinKMeans(n_clusters=3,n_init=2).fit(x)
    row=m.compare_refits(a,b,x,shape_only=False)
    assert -1<=row['ari']<=1
    assert row['centroid_displacement'] is None
    assert row['mapping'] is None


def test_one_state_agreement_is_flagged():
    m=module()
    x=np.array([[0,0],[1,1],[10,10],[11,11]],float)
    a=WassersteinKMeans(n_clusters=2,n_init=2).fit(x)
    row=m.compare_refits(a,a,x[:2],shape_only=False)
    assert row['ari']==1
    assert row['both_single_state']


def test_holdout_requires_verified_development_before_execution(tmp_path,monkeypatch):
    m=module()
    import yaml
    config=tmp_path/'asset.yaml'
    config.write_text(yaml.safe_dump({'window_length':63,'symbol':'A'}))
    study=tmp_path/'study.yaml'
    study.write_text(yaml.safe_dump({'assets':{'A':str(config)}}))
    monkeypatch.setattr(m,'run',lambda *a,**k:pytest.fail('holdout opened before development validation'))
    with pytest.raises(ValueError,match='development'):
        m.run_cross_asset(study,output_root=tmp_path,stage='holdout')


def test_default_stage_is_development(tmp_path, monkeypatch):
    import json
    import sys
    from wasserstein_regimes.cli import main
    m = module()
    observed = []
    monkeypatch.setattr(m, 'run_cross_asset', lambda *a, **k: observed.append(k['stage']))
    monkeypatch.setattr(sys, 'argv', ['regimes', 'cross-asset', '--config', 'study.yaml'])
    main()
    assert observed == ['development']


def test_acquisition_metadata_checks_identity(tmp_path):
    import hashlib
    import json
    m = module()
    source = tmp_path/'prices.csv'
    source.write_text('date,close')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = dict(symbol='QQQ', sha256=digest, first_date='1999-03-10', retrieved_at='2026-09-21')
    source.with_suffix('.json').write_text(json.dumps(metadata))
    config = dict(dataset=str(source), symbol='QQQ', dataset_sha256=digest)
    assert m.acquisition_metadata(config) == metadata
    with pytest.raises(ValueError, match='Acquisition'):
        m.acquisition_metadata(dict(config, symbol='TLT'))
    source.write_text('changed')
    with pytest.raises(ValueError, match='Acquisition'):
        m.acquisition_metadata(config)
