# SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import pytest

from wasserstein_regimes.risk import causal_forecasts, forecast_losses, paired_comparisons


def example(n=10):
    q=np.arange(1,n+1,dtype=float)
    states=np.array([-1,-1]+[0]*(n-2))
    return q,states


def test_matured_targets_shrinkage_empty_state_and_ewma():
    q,s=example()
    s[5]=1
    out=causal_forecasts(q,s,s,k=3,window=3,horizon=2,min_history=3,prior_count=2,ewma_decay=.5)
    f=out['forecasts']
    assert np.isnan(f[:2]).all()
    # First origin 2 matures at 4: target (q[3]+q[4])/2 = 4.5.
    assert out['regime_count'][3]==0 and out['regime_count'][4]==1
    assert f[3,3]==2.5
    assert f[4,3]==pytest.approx((4.5+2*3)/3)
    assert f[5,3]==3.5 and out['regime_count'][5]==0  # unseen state
    assert f[2,1]==2 and f[3,1]==3 and f[4,1]==4
    assert f[4,2]==4
    # State 0 has origins 2,3,4; origin 5 belongs to state 1.
    assert out['regime_count'][7]==3
    assert out['methods']==['expanding','ewma','rolling','regime','volatility_state']


def test_future_changes_cannot_change_any_existing_forecast():
    q,s=example(20)
    args=dict(k=3,window=3,horizon=2,min_history=3,prior_count=2)
    full=causal_forecasts(q,s,s,**args)
    prefix=causal_forecasts(q[:11],s[:11],s[:11],**args)
    np.testing.assert_array_equal(full['forecasts'][:11],prefix['forecasts'])
    q[11:]*=1000;s[11:]=2
    changed=causal_forecasts(q,s,s,**args)
    np.testing.assert_array_equal(full['forecasts'][:11],changed['forecasts'][:11])


def test_qlike_zero_targets_and_proper_mean_minimum():
    y=np.array([0.,2.,4.])
    losses=forecast_losses(y,np.full((3,1),2.))
    np.testing.assert_allclose(losses['qlike'][:,0],np.log(2)+y/2)
    assert losses['qlike'].mean()<forecast_losses(y,np.full((3,1),3.))['qlike'].mean()
    np.testing.assert_array_equal(losses['mse'][:,0],(y-2)**2)
    for bad in (np.array([np.nan,2,4]),np.array([-1.,2,4])):
        with pytest.raises(ValueError): forecast_losses(bad,np.ones((3,1)))
    with pytest.raises(ValueError): forecast_losses(y,np.zeros((3,1)))


def test_zero_history_is_floored_but_targets_are_not_changed():
    q,s=example(8);q[:]=0
    out=causal_forecasts(q,s,s,k=3,window=3,min_history=3,horizon=2,forecast_floor=1e-8)
    assert (out['forecasts'][2:]==1e-8).all()
    assert out['floor_applied'][2:].all()
    assert (q==0).all()


@pytest.mark.parametrize('change',[dict(window=0),dict(horizon=True),dict(min_history=2),dict(prior_count=0),dict(ewma_decay=1),dict(forecast_floor=0),dict(k=True)])
def test_invalid_parameters_fail(change):
    q,s=example()
    args=dict(k=3,window=3,min_history=3,horizon=2)
    with pytest.raises(ValueError): causal_forecasts(q,s,s,**dict(args,**change))


def test_malformed_state_or_return_history_fails():
    q,s=example()
    args=dict(k=3,window=3,min_history=3)
    for bad in (s.astype(float),np.zeros(len(s),dtype=int),np.array([-1]*len(s)),s[:-1]):
        with pytest.raises(ValueError): causal_forecasts(q,bad,s,**args)
    q[4]=np.nan
    with pytest.raises(ValueError): causal_forecasts(q,s,s,**args)


def test_paired_stationary_intervals_reproducible_and_preserve_pairing():
    x=np.arange(30,dtype=float)
    losses={'qlike':np.column_stack([x+2,x]),'mse':np.column_stack([2*x+4,2*x])}
    a=paired_comparisons(losses,['ewma','regime'],blocks=[3,10],repeats=50,seed=7)
    assert a==paired_comparisons(losses,['ewma','regime'],blocks=[3,10],repeats=50,seed=7)
    for row in a:
        expected=-2 if row['loss']=='qlike' else -4
        assert row['mean_difference']==row['lower']==row['upper']==expected
    with pytest.raises(ValueError): paired_comparisons(losses,['ewma','regime'],blocks=[30],repeats=50,seed=7)
