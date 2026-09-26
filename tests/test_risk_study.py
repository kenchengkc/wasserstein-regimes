# SPDX-License-Identifier: GPL-3.0-only
from dataclasses import replace
import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest
import yaml

from test_frozen import bundle  # shared synthetic frozen-model fixture
from wasserstein_regimes.risk_study import risk_panel, validate_risk_config


@pytest.fixture
def study(bundle,tmp_path):
    from wasserstein_regimes.artifact_store import digest,write_json
    from wasserstein_regimes.data import load_csv
    from wasserstein_regimes.frozen import load_bundle
    days=xcals.get_calendar('XNYS').sessions_in_range('2020-10-01','2023-05-01').tz_localize(None)
    assets=[];panel={}
    rng=np.random.default_rng(14)
    for symbol in ['A','B']:
        path=tmp_path/f'{symbol}.csv'
        prices=100*np.exp(np.cumsum(rng.normal(0,.01,len(days))))
        pd.DataFrame({'Date':days,'Adj Close':prices}).to_csv(path,index=False)
        sha=digest(path)
        write_json(path.with_suffix('.json'),dict(symbol=symbol,sha256=sha))
        assets.append(dict(symbol=symbol,dataset=str(path),sha256=sha))
        panel[symbol]=load_csv(path,provider='fixture',price_column='Adj Close')
    c=dict(schema_version=1,assets=assets,provider='fixture',price_column='Adj Close',
        start='2023-01-01',end='2023-05-01',as_of='2023-05-02T00:00:00Z',horizon=2,
        min_history=10,prior_count=10,ewma_decay=.94,forecast_floor=1e-12,
        bootstrap_repeats=40,bootstrap_blocks=[2,5],bootstrap_seed=2026,exposure='exploratory_previously_inspected')
    config=tmp_path/'risk.yaml';config.write_text(yaml.safe_dump(c))
    contract,model=load_bundle(bundle[0])
    return bundle[0],config,c,contract,model,panel


def test_panel_target_timing_common_origins_and_pending_tail(study,monkeypatch):
    from wasserstein_regimes.sliced import SlicedWassersteinKMedoids
    _,_,c,contract,model,panel=study
    monkeypatch.setattr(SlicedWassersteinKMedoids,'fit',lambda *a,**k:pytest.fail('risk must not refit'))
    frame,summary=risk_panel(panel,c,contract,model)
    assert (frame.price_start>pd.Timestamp(contract['calibration_end'])).all()
    scored=frame[frame.scored]
    assert len(frame)-len(scored)==2
    assert summary['scored_origins']==len(scored) and summary['pending_targets']==2
    assert (scored.target_available_at>scored.available_at).all()
    assert frame.date.iloc[-1]==pd.Timestamp('2023-05-01')
    first=next(iter(panel.values()))
    q=np.mean(np.expm1(np.column_stack([s.returns for s in panel.values()])),axis=1)**2
    i=first.dates.get_loc(scored.date.iloc[0])
    assert scored.target.iloc[0]==pytest.approx(q[i+1:i+3].mean())
    for method in ['expanding','ewma','rolling','regime','volatility_state']:
        assert frame[method].notna().all()
        assert summary['metrics'][method]['origins']==len(scored)
    assert len(summary['comparisons'])==16
    assert summary['point_in_time_snapshot'] is False


def test_future_prefix_and_training_bins_unchanged(study):
    _,_,c,contract,model,panel=study
    first,summary=risk_panel(panel,c,contract,model)
    altered={}
    cut=pd.Timestamp('2023-03-01')
    for name,s in panel.items():
        r=s.returns.copy();r[s.dates>cut]*=20
        altered[name]=replace(s,returns=r)
    changed,after=risk_panel(altered,c,contract,model)
    cols=['date','available_at','regime','volatility_state','expanding','ewma','rolling','regime_count','volatility_count']
    pd.testing.assert_frame_equal(first.loc[first.date<=cut,cols],changed.loc[changed.date<=cut,cols])
    assert summary['volatility_boundaries']==after['volatility_boundaries']


def test_asof_excludes_unavailable_targets_without_future_forecast_access(study):
    _,_,c,contract,model,panel=study
    frame,_=risk_panel(panel,c,contract,model)
    asof=frame.available_at.iloc[-4]
    limited,summary=risk_panel(panel,dict(c,as_of=asof.isoformat()),contract,model)
    assert (limited.available_at<=asof).all()
    assert (limited.loc[limited.scored,'target_available_at']<=asof).all()
    assert summary['pending_targets']==2
    assert limited.loc[~limited.scored,'target'].isna().all()
    assert summary['excluded_unavailable_origins']==3


def test_bad_history_and_contract_rejected(study):
    _,_,c,contract,model,panel=study
    with pytest.raises(ValueError,match='order'): risk_panel(dict(reversed(list(panel.items()))),c,contract,model)
    s=panel['B'];r=s.returns.copy();r[30]=np.nan
    with pytest.raises(ValueError,match='complete'): risk_panel(dict(panel,B=replace(s,returns=r)),c,contract,model)
    available=s.available_at.to_numpy(copy=True);available[35]=available[37]
    with pytest.raises(ValueError,match='availability'):
        risk_panel(dict(panel,B=replace(s,available_at=available)),c,contract,model)
    with pytest.raises(ValueError,match='convention'):
        risk_panel(dict(panel,B=replace(s,return_basis='provided_returns')),c,contract,model)
    for change in (dict(extra=1),dict(start='2022-01-01'),dict(horizon=True),dict(exposure='untouched'),dict(bootstrap_blocks=[0]),dict(as_of='2023-05-01')):
        with pytest.raises(ValueError): validate_risk_config(dict(c,**change),contract)


def test_real_risk_job_sealed_resumed_and_content_bound(study,tmp_path):
    from wasserstein_regimes.artifact_store import verify
    from wasserstein_regimes.jobs import prepare_job,run_jobs,verify_request
    path,cfg,c,contract,model,panel=study
    spec=dict(name='fixture-risk',type='risk',bundle=str(path),config=str(cfg))
    request=prepare_job(spec)
    assert {'risk.py','risk_study.py'}<=set(request['identity']['sources'])
    first=run_jobs([spec],output_root=tmp_path/'jobs',workers=1,timeout=30)
    row=json.loads(first.read_text())['jobs'][0]
    output=Path(row['output'])
    verify(output,{'risk.json','forecasts.parquet','execution.json','task.json'})
    summary=json.loads((output/'risk.json').read_text())
    assert summary['summary']['scored_origins']>0
    second=run_jobs([spec],output_root=tmp_path/'jobs',workers=1,timeout=30)
    resumed=json.loads(second.read_text())['jobs'][0]
    assert resumed['status']=='cached' and resumed['output']==row['output']
    cfg.write_text(yaml.safe_dump(dict(c,prior_count=11)))
    assert prepare_job(spec)['id']!=request['id']
    with pytest.raises(ValueError,match='input identity'): verify_request(request)


def test_origin_end_keeps_available_later_target_observations(study):
    _,_,c,contract,model,panel=study
    full,_=risk_panel(panel,c,contract,model)
    shortened,summary=risk_panel(panel,dict(c,end='2023-04-21'),contract,model)
    expected=full[full.date<='2023-04-21'].reset_index(drop=True)
    pd.testing.assert_frame_equal(shortened,expected)
    assert summary['pending_targets']==0
    assert shortened.target_end.iloc[-1]==pd.Timestamp('2023-04-25')


def test_compressed_exchange_session_cannot_be_a_daily_observation(study):
    _,_,c,contract,model,panel=study
    compressed={}
    for name,s in panel.items():
        i=s.dates.get_loc('2023-04-17')
        keep=np.arange(len(s.returns))!=i
        r=s.returns.copy();r[i+1]+=r[i]
        starts=s.price_start.to_numpy(copy=True);starts[i+1]=starts[i]
        compressed[name]=replace(s,returns=r[keep],dates=s.dates[keep],available_at=s.available_at[keep],
                                 price_start=starts[keep],price_end=s.price_end[keep])
    with pytest.raises(ValueError,match='exchange sessions'):
        risk_panel(compressed,c,contract,model)


def test_close_return_cannot_be_available_before_exchange_close(study):
    _,_,c,contract,model,panel=study
    premature={name:replace(s,available_at=s.dates.tz_localize('UTC')+pd.Timedelta(hours=17)) for name,s in panel.items()}
    with pytest.raises(ValueError,match='exchange close'):
        risk_panel(premature,c,contract,model)
