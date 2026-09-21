import json
import sys
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from wasserstein_regimes.synthetic import _delays, _metric_record, _novelty_distances, _stationary_once, exact_moment_blocks, run_synthetic
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wasserstein_regimes.benchmark as transport_benchmark
from wasserstein_regimes.benchmark import run_benchmark


def test_exact_blocks_share_four_moments_but_differ_in_w2():
    blocks, truth = exact_moment_blocks(seed=7, n_blocks=6)
    assert blocks.shape == (6, 300)
    assert set(truth) == {0, 1}
    moments = np.array([[np.mean(block ** power) for power in range(1, 5)] for block in blocks])
    assert np.max(np.abs(moments - moments[0])) < 1e-11
    assert np.sqrt(np.mean((np.sort(blocks[0]) - np.sort(blocks[1])) ** 2)) > 0


def test_synthetic_suite_keeps_repeats_and_censored_delay_json_safe():
    result = run_synthetic({"synthetic_repeats": 2, "window_lengths": [21], "seed": 9, "n_init": 1})
    json.dumps(result, allow_nan=False)
    variance = result["recovery"]["variance"]["21"]
    assert variance["repetitions"] == 2
    assert variance["methods"]["w2"]["pure"]["ari"]["n"] == 2
    delay = variance["methods"]["w2"]["detection_delay"]
    assert delay["events"] == delay["detected"] + delay["censored"]
    assert result["stationary"]["21"]["methods"]["w2"]["forced_k"] == 2


def test_exact_suite_shows_moment_collapse_and_w2_recovery():
    result = run_synthetic({"synthetic_repeats": 1, "window_lengths": [21], "seed": 4, "n_init": 1})
    exact = result["exact_moments"]
    assert exact["max_first_four_moment_gap"] < 1e-10
    assert exact["w2_between_laws"] > 0
    assert exact["w2_test_ari"] == 1.0
    assert exact["moments_effective_clusters"] == 1
    assert exact["methods"]["w1"]["test_ari"] == 1.0
    assert exact["methods"]["volatility"]["effective_clusters"] == 1
    assert exact["methods"]["moments"]["test_ari"] == 0.0


def test_delay_uses_raw_switch_coordinate_and_censors_only_unseen_switches():
    ends = np.array([2268, 2272, 2276, 2280])
    truth = np.array([1, 1, 1, 1])
    predicted = np.array([0, 1, 1, 1])
    delay = _delays(ends, truth, predicted, test_start=1512, stride=4)
    assert delay["events"] == 1
    assert delay["detected"] == 1
    assert delay["observed_delays"] == [4]


def test_null_scores_have_measurable_dwell_and_heldout_calibration():
    result = _stationary_once(length=21, stride=4, n_init=1, seed=11)
    row = result["w2"]
    assert row["switch_frequency"] is not None
    assert row["mean_dwell_scored_windows"] > 1
    assert row["calibration_segment"] == "heldout_validation"
    assert row["validation_windows"] > 0


def test_single_truth_state_mixed_slice_does_not_claim_ari_or_nmi():
    record = _metric_record(np.ones(5, dtype=int), np.array([0, 0, 1, 1, 1]))
    assert record["ari"] is None
    assert record["nmi"] is None
    assert record["balanced_accuracy"] == 0.6


def test_novelty_uses_empirical_transport_for_gmm_and_w1():
    model = SimpleNamespace(centers_=np.array([[0., 0.], [4., 4.]]))
    sample = np.array([[0., 2.]])
    np.testing.assert_allclose(_novelty_distances(model, "gmm", sample), [[np.sqrt(2), np.sqrt(10)]])
    np.testing.assert_allclose(_novelty_distances(model, "w1", sample), [[1., 3.]])


def test_benchmark_verifies_all_kernel_outputs_before_timing(monkeypatch):
    monkeypatch.setenv("VECLIB_MAXIMUM_THREADS", "1")
    traced = []
    original = transport_benchmark._naive
    def tracked(samples, centers):
        traced.append(tracemalloc.is_tracing())
        return original(samples, centers)
    monkeypatch.setattr(transport_benchmark, "_naive", tracked)
    result = run_benchmark(n=20, k=3, length=7, repeats=3)
    assert result["matrix"]["dtype"] == "float64"
    assert result["blas_threads"] == 1
    assert result["single_thread_control"]["accelerate_env"] == "1"
    assert result["machine"]["blas"]
    assert len(result["implementations"]) == 4
    assert all(row["max_absolute_error"] < 1e-12 for row in result["implementations"].values())
    assert traced.count(True) == 1
    assert len(result["implementations"]["naive_scalar_python"]["seconds"]) == 3
