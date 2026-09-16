import json
from pathlib import Path

import numpy as np
import pandas as pd


def _artifacts(path: Path):
    path.mkdir()
    config = dict(dataset='private <spy>.csv', provider='example & co', window_length=4,
                  fit_stride=2, score_stride=1, novelty_threshold=.99, horizon=21,
                  k_candidates=[2, 3], seed=42, models=['w2', 'shape_w2'])
    manifest = dict(run_id='sample-run', dataset_sha256='abc123', return_basis='adjusted-close log return',
                    point_in_time=False, quality={'missing_sessions': 0}, dependency_versions={'numpy': 'test'},
                    k=[3, 2], split_boundaries=[])
    outcomes = {name: dict(mean=value, n=3, block_interval=[value-.01, value+.01])
                for name, value in [('forward_log_return', .03), ('forward_volatility', .18),
                                    ('forward_drawdown', .08), ('forward_downside', .12)]}
    strict = dict(n=4, occupancy=[.75, .25], within_w2=.12, between_centroid_w2_mean=.32,
                  centroid_decomposition=dict(location=.2, scale=.3, shape=.5),
                  assignment_decomposition=dict(location=.1, scale=.7, shape=.2),
                  temporal=dict(mean_dwell=2.5, switching_frequency=.2), ood_rate=.25,
                  conditional_outcomes=[dict(regime=i, n=3-i, occupancy=[.75,.25][i],
                                             decomposition=dict(location=.1, scale=.7, shape=.2),
                                             **outcomes) for i in range(2)])
    other = dict(strict, within_w2=.8, distance_geometry='w2')
    fold_2019 = dict(year=2019, boundaries=dict(train_end='2015-12-31', validation_end='2018-12-31', test_end='2019-12-31'),
                     k=3, selection=[dict(k=2, validation_silhouette=.2), dict(k=3, validation_silhouette=.3)],
                     models={'w2': dict(strict=strict)}, ari_against_raw_w2={'w2': 1.0})
    fold_2020 = dict(year=2020, boundaries=dict(train_end='2016-12-31', validation_end='2019-12-31', test_end='2020-12-31'),
                     k=2, selection=[dict(k=2, validation_silhouette=.4), dict(k=3, validation_silhouette=.1)],
                     models={'w2': dict(strict=strict, distance_geometry='w2', shape_standardized=False),
                             'shape_w2': dict(strict=other, distance_geometry='w2', shape_standardized=True)},
                     ari_against_raw_w2={'w2': 1.0, 'shape_w2': .45})
    metrics = dict(folds=[fold_2019, fold_2020],
                   stability=dict(scoring_period='validation only', block_length=12, models={
                       name: {'seed': dict(ari_mean=.7, ari_range=[.6,.8], repetitions=3),
                              'block_bootstrap': dict(ari_mean=.4, ari_range=[.2,.6], repetitions=4)}
                       for name in ('w2','shape_w2','volatility','moments')}),
                   overlap=dict(period='validation only', null='IID resampling', rows=[
                       dict(fit_stride=2, observed_mean_dwell=4.0, null_mean_dwell=3.0,
                            null_interval=[2.5,3.5], ari_vs_primary=.9)]))
    for name, value in [('config', config), ('manifest', manifest), ('metrics', metrics)]:
        (path / f'{name}.json').write_text(json.dumps(value))
    np.savez_compressed(path / 'centroids.npz', centers=np.array([[-.04,-.02,.0,.01],[-.01,.0,.01,.04]]))
    rows = []
    for fold in (2019, 2020):
        for model in ('w2', 'shape_w2'):
            for policy in ('strict', 'operational'):
                for i in range(4):
                    rows.append(dict(fold=fold, model=model, policy=policy,
                                     date=pd.Timestamp('2020-01-02') + pd.Timedelta(days=i),
                                     regime=i % 2, novelty_percentile=[.2,.5,.99,.7][i],
                                     nearest_distance=.1+i*.01, ood=i == 2,
                                     w2_total=.1, w2_location=.01, w2_scale=.07, w2_shape=.02))
    pd.DataFrame(rows).to_parquet(path / 'assignments.parquet', index=False)


def test_report_uses_latest_strict_saved_results_and_escapes_input(tmp_path, monkeypatch):
    _artifacts(tmp_path / 'run')
    from wasserstein_regimes import clustering, baselines
    def forbidden(*args, **kwargs):
        raise AssertionError('report must not fit estimators')
    monkeypatch.setattr(clustering.WassersteinKMeans, 'fit', forbidden)
    monkeypatch.setattr(baselines.FeatureBaseline, 'fit', forbidden)
    from wasserstein_regimes.reporting import report
    path = report(tmp_path / 'run')
    html = path.read_text()
    assert path == tmp_path / 'run' / 'report.html'
    assert 'sample-run' in html and '2020 strict test fold' in html
    assert '2019' in html and 'selected K changed' in html
    assert 'Latest fold K candidates' in html and '0.100' in html
    assert 'switching frequency' in html and '2.500' in html
    assert 'private &lt;spy&gt;.csv' in html and 'example &amp; co' in html
    assert 'private <spy>.csv' not in html
    assert 'shape W2' in html and 'not comparable' in html
    assert '99th percentile' in html and 'not a probability' in html
    assert 'data:image/png;base64,' in html
    assert len(list((path.parent / 'figures').glob('*.png'))) >= 6


def test_report_is_reproducible_and_optional_sections_are_measured(tmp_path):
    _artifacts(tmp_path / 'run')
    run = tmp_path / 'run'
    (run / 'synthetic.json').write_text(json.dumps({'recovery_ari': .42, 'detection_delay': 8}))
    (run / 'benchmark.json').write_text(json.dumps({'runtime_seconds': 1.25}))
    from wasserstein_regimes.reporting import report
    first = report(run).read_bytes()
    second = report(run).read_bytes()
    assert first == second
    assert b'Synthetic recovery' in first and b'0.42' in first
    assert b'Benchmark' in first and b'1.25' in first


def test_synthetic_recovery_schema_gets_recovery_and_delay_figures(tmp_path):
    _artifacts(tmp_path / 'run')
    run = tmp_path / 'run'
    summary = lambda mean: {'n': 4, 'mean': mean, 'range': [mean-.1, mean+.1], 'standard_error': .05}
    synthetic = {'config': {'synthetic_repeats': 4}, 'recovery': {
        'normal_t5': {'63': {'repetitions': 4, 'score_stride': 12, 'methods': {
            'w2': {'pure': {'ari': summary(.65)}, 'mixed': {'ari': summary(.4)},
                   'detection_delay': {'detected_only_delay': summary(7), 'events': 8,
                                       'detected': 6, 'censored': 2, 'failure_rate': .25}},
            'moments': {'pure': {'ari': summary(.25)}, 'mixed': {'ari': summary(.1)},
                        'detection_delay': {'detected_only_delay': summary(13), 'events': 8,
                                            'detected': 5, 'censored': 3, 'failure_rate': .375}}
        }}}
    }, 'inference_note': 'independent streams'}
    (run / 'synthetic.json').write_text(json.dumps(synthetic))
    from wasserstein_regimes.reporting import report
    html = report(run).read_text()
    assert 'normal_t5' in html and '0.650' in html and '7.000' in html
    assert 'censored' in html and 'detected-only' in html
    assert (run/'figures'/'synthetic_recovery.png').exists()
    assert (run/'figures'/'synthetic_delay.png').exists()
