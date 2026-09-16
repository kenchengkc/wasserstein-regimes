# SPDX-License-Identifier: GPL-3.0-only
"""Measured float64 transport kernels on one shared sorted matrix."""

from __future__ import annotations

import os
import platform
import subprocess
import time
import tracemalloc

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from wasserstein_regimes.transport import _pairwise_sorted_costs


def _naive(samples, centers):
    out = np.empty((len(samples), len(centers)), dtype=np.float64)
    for i, row in enumerate(samples):
        for j, center in enumerate(centers):
            total = 0.0
            for a, b in zip(row, center):
                delta = float(a - b)
                total += delta * delta
            out[i, j] = total / samples.shape[1]
    return out


def _broadcast(samples, centers):
    return np.mean((samples[:, None, :] - centers[None, :, :]) ** 2, axis=2)


def _blas(samples, centers):
    length = samples.shape[1]
    costs = (np.sum(samples ** 2, axis=1)[:, None]
             + np.sum(centers ** 2, axis=1)[None, :]
             - 2 * samples @ centers.T) / length
    np.maximum(costs, 0, out=costs)
    return costs


def _machine():
    def sysctl(name):
        try:
            return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    ram = sysctl("hw.memsize")
    return {"platform": platform.platform(), "cpu": sysctl("machdep.cpu.brand_string") or platform.processor(),
            "logical_cpus": os.cpu_count(), "ram_bytes": int(ram) if ram else None,
            "threadpools": threadpool_info(),
            "blas": np.__config__.CONFIG.get("Build Dependencies", {}).get("blas", {}).get("name")}


def run_benchmark(*, n=2000, k=5, length=63, repeats=3, seed=42) -> dict:
    """Warm each implementation, verify agreement, then measure time and allocations."""
    if min(n, k, length, repeats) < 1:
        raise ValueError("benchmark dimensions and repeats must be positive")
    rng = np.random.default_rng(seed)
    samples = np.sort(rng.normal(size=(n, length)).astype(np.float64), axis=1)
    centers = np.sort(rng.normal(size=(k, length)).astype(np.float64), axis=1)
    implementations = {"naive_scalar_python": _naive,
                       "broadcast_numpy": _broadcast,
                       "chunked_numpy": lambda a, b: _pairwise_sorted_costs(a, b, 2, 256),
                       "matrix_product_blas": _blas}
    results = {}
    with threadpool_limits(limits=1):
        reference = implementations["naive_scalar_python"](samples, centers)
        for name, implementation in implementations.items():
            warmed = implementation(samples, centers)
            error = float(np.max(np.abs(warmed - reference)))
            if not np.allclose(warmed, reference, rtol=1e-10, atol=1e-12):
                raise AssertionError(f"{name} disagrees with scalar reference: max error {error}")
            times = []
            for _ in range(repeats):
                start = time.perf_counter()
                actual = implementation(samples, centers)
                times.append(time.perf_counter() - start)
                if not np.allclose(actual, reference, rtol=1e-10, atol=1e-12):
                    raise AssertionError(f"{name} became numerically inconsistent")
            tracemalloc.start()
            allocated = implementation(samples, centers)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            if not np.allclose(allocated, reference, rtol=1e-10, atol=1e-12):
                raise AssertionError(f"{name} became numerically inconsistent in allocation pass")
            results[name] = {"seconds": times, "median_seconds": float(np.median(times)),
                             "peak_tracemalloc_bytes": int(peak), "max_absolute_error": error}
    return {"matrix": {"n": n, "k": k, "length": length, "dtype": "float64", "sorted": True},
            "repeats": repeats, "warmup": True, "blas_threads": 1,
            "timing_tracemalloc_active": False,
            "single_thread_control": {"threadpoolctl_requested": 1,
                                      "accelerate_env": os.getenv("VECLIB_MAXIMUM_THREADS"),
                                      "threadpoolctl_visibility": bool(threadpool_info())},
            "memory_note": "tracemalloc peak Python-traced allocations, not process RSS",
            "machine": _machine(), "implementations": results}


if __name__ == "__main__":
    import json
    print(json.dumps(run_benchmark(), indent=2))
