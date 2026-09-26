# SPDX-License-Identifier: GPL-3.0-only
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_complete_inventory_rejects_tampering_omissions_and_extra_files(tmp_path):
    from wasserstein_regimes.artifact_store import seal, verify
    (tmp_path/'a.json').write_text('{}')
    seal(tmp_path)
    verify(tmp_path, {'a.json'})
    (tmp_path/'extra').write_text('x')
    with pytest.raises(ValueError): verify(tmp_path, {'a.json'})
    (tmp_path/'extra').unlink()
    (tmp_path/'a.json').write_text('changed')
    with pytest.raises(ValueError): verify(tmp_path, {'a.json'})
    (tmp_path/'checksums.json').write_text('{}')
    with pytest.raises(ValueError): verify(tmp_path, {'a.json'})


@pytest.fixture
def bundle(tmp_path):
    from wasserstein_regimes.artifact_store import seal, write_json
    from wasserstein_regimes.experiments import code_provenance
    from wasserstein_regimes.frozen import freeze_joint, numerical_sources
    from wasserstein_regimes.sliced import SlicedWassersteinKMedoids
    from wasserstein_regimes.joint_market import source_identity
    dev=tmp_path/'development'
    (dev/'models').mkdir(parents=True)
    x=np.random.default_rng(8).normal(size=(30,5,2))
    model=SlicedWassersteinKMedoids(n_clusters=2,n_projections=4,n_init=1,candidate_size=12).fit(x)
    model.save(dev/'models/scaled_joint.npz')
    write_json(dev/'config.json',dict(assets=[dict(symbol='A'),dict(symbol='B')],window_length=5,
        provider='fixture',price_column='Adj Close',train_end='2020-12-31',validation_end='2022-12-31'))
    write_json(dev/'manifest.json',dict(stage='development',symbol_order=['A','B'],
        source_files=source_identity(),dependency_versions=code_provenance()['dependency_versions']))
    write_json(dev/'metrics.json',dict(fit=dict(k=2),calibration=dict(scaled_joint=2.)))
    seal(dev)
    path=freeze_joint(dev,output_root=tmp_path/'bundles')
    return path, model


def test_bundle_round_trip_and_original_bytes(bundle,tmp_path):
    from wasserstein_regimes.frozen import load_bundle, freeze_joint
    from wasserstein_regimes.artifact_store import verify
    path, original=bundle
    contract, model=load_bundle(path)
    assert contract['symbols']==['A','B'] and contract['window_length']==5
    assert (path/'model.npz').read_bytes()==(tmp_path/'development/models/scaled_joint.npz').read_bytes()
    x=np.random.default_rng(2).normal(size=(8,5,2))
    np.testing.assert_array_equal(model.predict(x),original.predict(x))
    assert freeze_joint(tmp_path/'development',output_root=tmp_path/'bundles')==path
    verify(path, {'contract.json','model.npz'})
    (path/'model.npz').write_bytes(b'bad')
    with pytest.raises(ValueError): load_bundle(path)


def test_score_contract_order_availability_and_no_fitting(bundle,monkeypatch):
    from wasserstein_regimes.data import ReturnSeries
    from wasserstein_regimes.frozen import score_panel, load_bundle
    from wasserstein_regimes.sliced import SlicedWassersteinKMedoids
    path,original=bundle
    contract,model=load_bundle(path)
    dates=pd.bdate_range('2020-12-28','2021-02-05')
    rng=np.random.default_rng(3)
    def series():
        return ReturnSeries(rng.normal(size=len(dates)-1),dates[1:],dates[1:].tz_localize('UTC')+pd.Timedelta(hours=21),
                            dates[:-1],dates[1:],provider='fixture',return_basis='adjusted_close')
    panel={'A':series(),'B':series()}
    monkeypatch.setattr(SlicedWassersteinKMedoids,'fit',lambda *a,**k:pytest.fail('scoring fitted'))
    args=dict(start='2021-01-01',end='2021-02-05',as_of='2021-02-05T20:00:00Z')
    frame,summary=score_panel(panel,contract,model,**args)
    assert frame.price_start.min()>pd.Timestamp(contract['train_end'])
    assert frame.date.max()==pd.Timestamp('2021-02-04')
    assert summary['excluded_unavailable']==1
    assert summary['novelty_interpretation']=='retrospective_calibration_reuse'
    assert len(frame)==sum(summary['counts'])
    with pytest.raises(ValueError,match='order'):
        score_panel(dict(reversed(list(panel.items()))),contract,model,**args)
    with pytest.raises(ValueError,match='timezone'):
        score_panel(panel,contract,model,**dict(args,as_of='2021-02-05'))
    wrong=dict(panel,B=replace(panel['B'],provider='other'))
    with pytest.raises(ValueError,match='convention'): score_panel(wrong,contract,model,**args)


def test_contract_unknown_fields_and_basis_rejected(bundle):
    from wasserstein_regimes.frozen import validate_contract,load_bundle
    c,_=load_bundle(bundle[0])
    for change in [dict(extra=4),dict(return_basis='simple_returns'),dict(symbols=['A','A']),dict(schema_version=True)]:
        with pytest.raises(ValueError): validate_contract(dict(c,**change))


def test_freeze_rejects_parent_changed_during_copy(bundle,tmp_path,monkeypatch):
    from wasserstein_regimes import frozen
    from wasserstein_regimes.artifact_store import seal
    original=frozen.shutil.copyfile
    dev=tmp_path/'development'
    def changed(source,target):
        result=original(source,target)
        metrics=json.loads((dev/'metrics.json').read_text())
        metrics['calibration']['scaled_joint']=3.
        (dev/'metrics.json').write_text(json.dumps(metrics))
        seal(dev)
        return result
    monkeypatch.setattr(frozen.shutil,'copyfile',changed)
    with pytest.raises(ValueError,match='changed'):
        frozen.freeze_joint(dev,output_root=tmp_path/'new')


def test_freeze_rejects_parent_changed_before_digest_capture(bundle,tmp_path,monkeypatch):
    from wasserstein_regimes import frozen
    from wasserstein_regimes.artifact_store import seal
    original=Path.read_text
    metrics=tmp_path/'development/metrics.json'
    changed=False
    def read(path,*args,**kwargs):
        nonlocal changed
        value=original(path,*args,**kwargs)
        if path==metrics and not changed:
            changed=True
            latest=json.loads(value)
            latest['calibration']['scaled_joint']=3.
            metrics.write_text(json.dumps(latest))
            seal(metrics.parent)
        return value
    monkeypatch.setattr(Path,'read_text',read)
    with pytest.raises(ValueError,match='changed'):
        frozen.freeze_joint(metrics.parent,output_root=tmp_path/'new')


def test_job_identity_tracks_consumed_sidecar_and_symlink_retarget(bundle,tmp_path):
    import yaml
    from wasserstein_regimes.jobs import prepare_job,verify_request
    from wasserstein_regimes.artifact_store import digest
    source=tmp_path/'data.csv';source.write_text('original snapshot')
    source.with_suffix('.json').write_text('{"not_consumed":true}')
    alias=tmp_path/'alias.csv';alias.symlink_to(source)
    alias.with_suffix('.json').write_text('{"consumed":true}')
    second=tmp_path/'B.csv';second.write_text('other snapshot')
    second.with_suffix('.json').write_text('{}')
    cfg=tmp_path/'score.yaml'
    cfg.write_text(yaml.safe_dump(dict(schema_version=1,provider='fixture',price_column='Adj Close',
        start='2021-01-01',end='2021-02-01',as_of='2021-02-02T00:00:00Z',assets=[
            dict(symbol='A',dataset=str(alias),sha256=digest(alias)),dict(symbol='B',dataset=str(second),sha256=digest(second))])))
    spec=dict(name='score',type='score',bundle=str(bundle[0]),config=str(cfg))
    request=prepare_job(spec)
    assert str(alias.with_suffix('.json')) in request['identity']['inputs']
    alias.with_suffix('.json').write_text('{"consumed":"changed"}')
    assert prepare_job(spec)['id']!=request['id']
    with pytest.raises(ValueError,match='input identity'): verify_request(request)
    request=prepare_job(spec)
    alternate=tmp_path/'alternate.csv';alternate.write_text('different data')
    alias.unlink();alias.symlink_to(alternate)
    with pytest.raises(ValueError,match='input identity'): verify_request(request)
