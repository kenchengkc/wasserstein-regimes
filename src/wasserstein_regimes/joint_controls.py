# SPDX-License-Identifier: GPL-3.0-only
"""Frozen dependence nulls and rare/gradual joint-return recovery experiments."""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

from .joint_market import _directions
from .panel_baselines import PanelKMeans, panel_features
from .robustness import control_stream, partition_diagnostics, pure_truth, rare_delay, recovery_metrics, return_windows
from .sliced import SlicedWassersteinKMedoids, sliced_w2


def covariance_features(model, samples):
    return (panel_features(samples/model.scales_, 'covariance') - model.feature_mean_) / model.feature_std_


def _distances(model, samples):
    if isinstance(model, SlicedWassersteinKMedoids):
        return model.transform(samples)
    return cdist(covariance_features(model, samples), model.centers_)


def _silhouette(model, samples, labels, seed):
    indices = np.sort(np.random.default_rng(seed).choice(len(samples), min(160, len(samples)), replace=False))
    labels = labels[indices]
    if not 1 < len(np.unique(labels)) < len(labels):
        return None
    if isinstance(model, SlicedWassersteinKMedoids):
        distances = sliced_w2(samples[indices], samples[indices], model.projections_, scales=model.scales_)
        np.fill_diagonal(distances, 0.)
        return float(silhouette_score(distances, labels, metric='precomputed'))
    return float(silhouette_score(covariance_features(model, samples[indices]), labels))


def run_controls(c):
    rows = []
    directions = _directions(5, 64, 42)
    with threadpool_limits(limits=1):
        for kind in ('gaussian_null', 'student_null', 'rare', 'gradual'):
            for seed in c['synthetic_seeds']:
                values, rho = control_stream(kind, seed)
                train_all = return_windows(values[:1500], 63)
                train = train_all[::5]
                calibration = return_windows(values[1500:2250], 63)
                test = return_windows(values[2250:], 63)
                scales = values[:1500].std(axis=0)
                null = kind.endswith('null')
                budgets = [128] if null else c['candidate_sizes']
                for budget in budgets:
                    print(f'synthetic {kind} seed={seed} candidates={budget}', flush=True)
                    # Covariance is rerun for each row to keep a complete matched comparison.
                    for method in ('joint', 'covariance'):
                        k = 3 if null else 2
                        if method == 'joint':
                            model = SlicedWassersteinKMedoids(n_clusters=k, candidate_size=budget, n_init=5,
                                random_state=seed, scales=scales, projections=directions).fit(train)
                        else:
                            model = PanelKMeans('covariance', k, scales=scales, n_init=5, seed=seed).fit(train)
                        labels = model.predict(test)
                        row = dict(control=kind, seed=seed, candidates=budget, method=method, forced_k=k,
                                   training_windows=len(train), calibration_windows=len(calibration),
                                   test=partition_diagnostics(labels, k))
                        if method == 'joint':
                            row.update(converged=model.converged_,
                                       candidate_converged=[run.get('converged', False) for run in model.candidate_runs_])
                        if null:
                            threshold = float(np.quantile(_distances(model, calibration).min(axis=1), .99))
                            row.update(novelty_threshold=threshold,
                                       novelty_fraction=float(np.mean(_distances(model, test).min(axis=1) > threshold)),
                                       test_silhouette=_silhouette(model, test, labels, seed),
                                       distance='sliced_w2' if method == 'joint' else 'standardized_covariance_euclidean')
                        else:
                            recovery = recovery_metrics(pure_truth(rho[:1500], 63), model.predict(train_all),
                                                        pure_truth(rho[2250:], 63), labels)
                            row['recovery'] = recovery
                            if kind == 'rare':
                                row['delay'] = rare_delay(np.arange(2250+62, 3000), np.array(recovery['mapping'])[labels])
                        rows.append(row)
    return rows
