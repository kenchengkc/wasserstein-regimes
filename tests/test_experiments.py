import importlib.util
import numpy as np
import pandas as pd


def module():
    assert importlib.util.find_spec('wasserstein_regimes.experiments') is not None
    from wasserstein_regimes import experiments
    return experiments


def test_split_purges_shared_preceding_price():
    e = module()
    dates = pd.date_range('2020-01-01', periods=12)
    starts = dates - pd.Timedelta(days=3)
    masks = e.split_masks(dates, starts, train_end='2020-01-04', validation_end='2020-01-08', test_end='2020-01-12', strict=True)
    assert np.flatnonzero(masks['train']).tolist() == [0,1,2,3]
    assert np.flatnonzero(masks['validation']).tolist() == [7]
    assert np.flatnonzero(masks['test']).tolist() == [11]
    op = e.split_masks(dates, starts, train_end='2020-01-04', validation_end='2020-01-08', test_end='2020-01-12', strict=False)
    assert np.flatnonzero(op['test']).tolist() == [8,9,10,11]


def test_artifact_identity_order_independent_but_content_sensitive():
    e = module()
    a = e.artifact_id({'a':1,'b':2}, {'sha':'x'})
    assert a == e.artifact_id({'b':2,'a':1}, {'sha':'x'})
    assert a != e.artifact_id({'a':2,'b':2}, {'sha':'x'})


def test_prefix_model_and_predictions_do_not_use_future_observations():
    e = module()
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 8))
    changed = x.copy()
    changed[60:] *= 100
    a = e.fit_primary(x[:40], x[40:60], k_candidates=[2,3], seed=7, n_init=2)
    b = e.fit_primary(changed[:40], changed[40:60], k_candidates=[2,3], seed=7, n_init=2)
    np.testing.assert_array_equal(a[0].centers_, b[0].centers_)
    np.testing.assert_array_equal(a[0].predict(x[:60]), b[0].predict(changed[:60]))
    assert a[1] == b[1]


def test_csv_pipeline_prefix_invariance_and_strict_intervals(tmp_path):
    e = module()
    from wasserstein_regimes.data import from_prices
    from wasserstein_regimes.windows import make_windows
    import exchange_calendars as xc
    dates = xc.get_calendar('XNYS', start='2015-01-01', end='2020-12-31').sessions
    rng = np.random.default_rng(19)
    prices = 100*np.exp(np.cumsum(rng.normal(0,.01,len(dates))))
    cutoff = pd.Timestamp('2019-06-28')
    altered = prices.copy()
    altered[dates > cutoff] *= np.exp(np.arange(sum(dates > cutoff)) * .3)
    config = dict(validation_years=1, fit_stride=5, k_candidates=[2], seed=42,n_init=2,
                  models=['w2'],window_length=21,novelty_threshold=.99,horizon=5)
    results=[]
    for i,p in enumerate((prices,altered)):
        series=from_prices(pd.DataFrame({'Date':dates,'Adj Close':p}))
        batch=make_windows(series,length=21)
        summary, frame, models=e.run_fold(series,batch,config,year=2019,test_end='2019-06-28',output=tmp_path/str(i))
        results.append((summary,frame,models['w2']))
        strict=frame[frame.policy=='strict']
        assert (strict.price_start > pd.Timestamp('2018-12-31')).all()
    np.testing.assert_array_equal(results[0][2].centers_, results[1][2].centers_)
    columns=['date','regime','nearest_distance','novelty_percentile','ood']
    pd.testing.assert_frame_equal(results[0][1][columns],results[1][1][columns])
    # Future evaluation outcomes may change; fitted state and available predictions may not.


def test_completed_artifact_rejects_modified_assignments(tmp_path):
    e=module()
    (tmp_path/'assignments.parquet').write_bytes(b'original')
    (tmp_path/'metrics.json').write_text('{}')
    e.seal_artifact(tmp_path)
    e.verify_artifact(tmp_path)
    (tmp_path/'assignments.parquet').write_bytes(b'changed')
    import pytest
    with pytest.raises(ValueError,match='integrity'):
        e.verify_artifact(tmp_path)
