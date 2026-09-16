import importlib.util
import numpy as np
import pandas as pd
from types import SimpleNamespace


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
    assert (tmp_path/'0'/'model.json').read_bytes() == (tmp_path/'1'/'model.json').read_bytes()
    with np.load(tmp_path/'0'/'model.npz', allow_pickle=False) as left, np.load(tmp_path/'1'/'model.npz', allow_pickle=False) as right:
        for name in left.files:
            np.testing.assert_array_equal(left[name], right[name])
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


def test_stability_uses_effective_reference_count_for_collapsed_volatility():
    e = module()
    a = np.array([-1., -1., 1., 1.])
    b = np.array([-np.sqrt(2), 0., 0., np.sqrt(2)])
    train = np.tile(np.stack([a, b]), (12, 1))
    validation = np.tile(np.stack([a, b]), (6, 1))
    dates = pd.DatetimeIndex(list(pd.date_range('2017-06-01', periods=len(train))) +
                             list(pd.date_range('2018-06-01', periods=len(validation))))
    batch = SimpleNamespace(samples=np.vstack([train, validation]), dates=dates,
                            price_start=dates - pd.Timedelta(days=1))
    raw_dates = pd.date_range('2017-01-01', periods=80)
    series = SimpleNamespace(returns=np.random.default_rng(9).normal(size=len(raw_dates)), dates=raw_dates)
    config = dict(validation_years=1, fit_stride=1, k_candidates=[2], seed=5, n_init=2,
                  bootstrap_block_length=4, window_length=4, seed_repeats=1, bootstrap_repeats=1)

    result = e.stability_study(series, batch, config, year=2019, test_end='2019-06-30')
    volatility = result['models']['volatility']
    seed = volatility['seed']
    bootstrap = volatility['block_bootstrap']
    assert seed['reference_effective_clusters'] == 1
    assert seed['draws'][0]['effective_clusters'] == 1
    assert seed['draws'][0]['centroid_displacement'] is not None
    assert seed['draws'][0]['occupancy_l1'] == 0.
    assert bootstrap['reference_effective_clusters'] == 1
    assert bootstrap['draws'][0]['effective_clusters'] == 2
    assert bootstrap['draws'][0]['centroid_displacement'] is None
    assert bootstrap['draws'][0]['occupancy_l1'] is None


def test_single_prototype_summary_keeps_novelty_and_occupancy():
    e = module()
    from wasserstein_regimes.windows import WindowBatch
    samples = np.tile([-1., -1., 1., 1.], (4, 1))
    dates = pd.date_range('2019-01-01', periods=4)
    endpoints = np.array([3, 7, 11, 15])
    batch = WindowBatch(samples, endpoints, endpoints - 3, dates, dates.tz_localize('UTC'),
                        dates - pd.Timedelta(days=1), dates)
    frame, metrics, _ = e._summary(batch, np.zeros(4, dtype=int), samples[:1],
                                    np.zeros((4, 1)), [0., 1.], np.tile(samples[0], 5),
                                    threshold=.99, horizon=1, bandwidth=1., seed=5)
    np.testing.assert_array_equal(frame.nearest_distance, [0., 0., 0., 0.])
    np.testing.assert_array_equal(frame.novelty_percentile, [.5, .5, .5, .5])
    assert frame.second_distance.isna().all()
    assert frame.margin.isna().all()
    assert not frame.ood.any()
    assert metrics['occupancy'] == [1.]
    assert metrics['between_centroid_w2_mean'] is None
    assert metrics['temporal']['transitions'] == [[0]]


def test_run_root_model_is_loadable_and_matches_latest_fold(tmp_path, monkeypatch):
    import hashlib
    import yaml
    import exchange_calendars as xc
    from wasserstein_regimes.clustering import WassersteinKMeans
    from wasserstein_regimes import synthetic, benchmark
    e = module()
    dates = xc.get_calendar('XNYS', start='2015-01-01', end='2019-12-31').sessions
    prices = 100 * np.exp(np.cumsum(np.random.default_rng(19).normal(0, .01, len(dates))))
    source = tmp_path / 'prices.csv'
    pd.DataFrame({'Date': dates, 'Adj Close': prices}).to_csv(source, index=False)
    config = dict(dataset=str(source), dataset_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  price_column='Adj Close', provider='test fixture', data_cutoff='2019-12-31',
                  holdout_start='2020-01-01', development_years=[2019], window_length=21,
                  score_stride=1, fit_stride=5, metric='w2', validation_years=1,
                  k_candidates=[2], seed=42, n_init=2, models=['w2'], novelty_threshold=.99, horizon=5)
    spec = tmp_path / 'study.yaml'
    spec.write_text(yaml.safe_dump(config))
    # These supplementary studies do not participate in fold model persistence.
    monkeypatch.setattr(e, 'stability_study', lambda *args, **kwargs: {})
    monkeypatch.setattr(e, 'overlap_study', lambda *args, **kwargs: {})
    monkeypatch.setattr(synthetic, 'run_synthetic', lambda config: {})
    monkeypatch.setattr(benchmark, 'run_benchmark', lambda: {})

    output = e.run(spec, output_root=tmp_path / 'artifacts')
    root = WassersteinKMeans.load(output / 'model')
    latest = WassersteinKMeans.load(output / 'models' / '2019' / 'model')
    np.testing.assert_array_equal(root.centers_, latest.centers_)
    np.testing.assert_array_equal(root.predict(latest.centers_), [0, 1])
    with np.load(output / 'centroids.npz', allow_pickle=False) as centroids:
        np.testing.assert_array_equal(root.centers_, centroids['centers'])
    e.verify_artifact(output)
