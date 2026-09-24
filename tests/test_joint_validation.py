# SPDX-License-Identifier: GPL-3.0-only
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


def config():
    return yaml.safe_load(Path('configs/joint_validation.yaml').read_text())


def test_strict_validation_schema():
    from wasserstein_regimes.joint_validation import validate_config
    validate_config(config())
    for changes in [dict(extra=3), dict(seed=True), dict(bootstrap_repeats=0),
                    dict(candidate_sizes=[1]), dict(outlier_fractions=[float('nan')]),
                    dict(chronological_ends=['2020-01-01', '2019-01-01']),
                    dict(optimizer_seeds=[42, 42]), dict(block_lengths=[]), dict(market_config=None)]:
        with pytest.raises(ValueError): validate_config(dict(config(), **changes))


def test_refit_recomputes_unique_return_scales_and_does_not_fit_anchors():
    from wasserstein_regimes.joint_validation import fit_returns
    from wasserstein_regimes.joint_market import _directions
    x = np.random.default_rng(6).normal(size=(120, 2))
    c = dict(window_length=7, fit_stride=3, candidate_size=16, n_init=2, seed=42, chunk_size=16)
    model = fit_returns(x, c, 2, _directions(2, 4, 42))
    np.testing.assert_allclose(model.scales_, x.std(axis=0))
    assert len(model.labels_) == 38


def test_synthetic_suite_uses_disjoint_segments_and_reports_every_cell(monkeypatch):
    from wasserstein_regimes.joint_controls import run_controls, PanelKMeans
    actual_fit = PanelKMeans.fit
    restarts = []
    def checked_fit(self, samples):
        restarts.append(self.n_init)
        return actual_fit(self, samples)
    monkeypatch.setattr(PanelKMeans, 'fit', checked_fit)
    c = dict(config(), synthetic_seeds=[17], candidate_sizes=[64])
    result = run_controls(c)
    assert restarts == [5, 5, 5, 5]
    assert len(result) == 8  # two methods x (two nulls + rare + gradual)
    assert all(row['test']['n'] == 688 for row in result)
    assert all(row['training_windows'] == 288 for row in result)
    for row in result:
        if row['control'].endswith('null'):
            assert 0 <= row['novelty_fraction'] <= 1 and row['calibration_windows'] == 688
        else:
            assert sum(row['recovery']['pure_counts']) + row['recovery']['mixed_windows'] == 688
            assert row['test']['n'] == sum(row['test']['counts'])


@pytest.fixture
def parent(tmp_path, monkeypatch):
    import wasserstein_regimes.joint_validation as module
    from wasserstein_regimes.data import ReturnSeries
    from wasserstein_regimes.experiments import artifact_id, seal_artifact, write_json, code_provenance
    from wasserstein_regimes.joint_market import prepare_panel, source_identity, _directions
    c = yaml.safe_load(Path('configs/joint_market.yaml').read_text())
    c.update(window_length=5, fit_stride=5, n_projections=4, candidate_size=12, n_init=1, k_candidates=[2])
    dates = pd.bdate_range('2018-01-01', '2023-12-29')
    rng = np.random.default_rng(7)
    base = ReturnSeries(rng.normal(size=len(dates)-1), dates[1:], dates[1:].tz_localize('UTC'), dates[:-1], dates[1:])
    series = {'SPY':base, 'QQQ':replace(base, returns=rng.normal(size=len(base.returns)))}
    c['assets'] = c['assets'][:2]
    metadata = {'fixture':'immutable'}
    def load(mc, cutoff):
        assert cutoff == mc['validation_end']
        panel, batch, masks = prepare_panel(series, mc, cutoff)
        return panel, batch, masks, metadata
    monkeypatch.setattr(module, 'load_panel', load)
    monkeypatch.setattr(module, 'run_controls', lambda c: [])
    dev = tmp_path/'development'
    (dev/'models').mkdir(parents=True)
    vectors = np.column_stack([v.returns[v.dates <= c['train_end']] for v in series.values()])
    model = module.fit_returns(vectors, c, 2, _directions(2, 4, 42))
    model.save(dev/'models/scaled_joint.npz')
    write_json(dev/'config.json', c)
    write_json(dev/'metrics.json', dict(fit=dict(k=2)))
    write_json(dev/'manifest.json', dict(stage='development', source_files=source_identity(),
        dependency_versions=code_provenance()['dependency_versions'], data_identity=artifact_id(metadata, {})))
    seal_artifact(dev)
    market = tmp_path/'market.yaml'
    market.write_text(yaml.safe_dump(c))
    vc = dict(config(), market_config=str(market), bootstrap_repeats=1, block_lengths=[5],
              chronological_ends=['2019-12-31'], candidate_sizes=[12], optimizer_seeds=[42], outlier_fractions=[.01])
    cfg = tmp_path/'validation.yaml'
    cfg.write_text(yaml.safe_dump(vc))
    return cfg, dev, metadata


def test_runner_seals_reuses_cache_and_checks_parent_provenance(parent, tmp_path, monkeypatch):
    import wasserstein_regimes.joint_validation as module
    from wasserstein_regimes.experiments import verify_artifact
    cfg, dev, metadata = parent
    report = module.run_joint_validation(cfg, development=dev, output_root=tmp_path)
    verify_artifact(report.parent)
    metrics = json.loads((report.parent/'metrics.json').read_text())
    assert len(metrics['market']) == 4
    assert len({row['n'] for row in metrics['market']}) == 1
    assert all(row['n'] == sum(row['counts']) for row in metrics['market'])
    monkeypatch.setattr(module, 'market_diagnostics', lambda *a: pytest.fail('cache refitted'))
    assert module.run_joint_validation(cfg, development=dev, output_root=tmp_path) == report
    metadata['revised'] = True
    with pytest.raises(ValueError, match='acquisition'):
        module.run_joint_validation(cfg, development=dev, output_root=tmp_path)
    del metadata['revised']
    (report.parent/'metrics.json').write_text('{}')
    with pytest.raises(ValueError, match='integrity'):
        module.run_joint_validation(cfg, development=dev, output_root=tmp_path)


def test_parent_missing_checksum_entry_and_changed_sources_rejected(parent, tmp_path):
    from wasserstein_regimes.joint_validation import run_joint_validation
    from wasserstein_regimes.experiments import seal_artifact
    cfg, dev, _ = parent
    manifest = json.loads((dev/'manifest.json').read_text())
    manifest['source_files']['sliced.py'] = 'changed'
    (dev/'manifest.json').write_text(json.dumps(manifest))
    seal_artifact(dev)
    with pytest.raises(ValueError, match='source'):
        run_joint_validation(cfg, development=dev, output_root=tmp_path)
    hashes = json.loads((dev/'checksums.json').read_text())
    del hashes['manifest.json']
    (dev/'checksums.json').write_text(json.dumps(hashes))
    with pytest.raises(ValueError, match='inventory'):
        run_joint_validation(cfg, development=dev, output_root=tmp_path)
