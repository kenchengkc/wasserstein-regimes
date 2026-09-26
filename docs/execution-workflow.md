# Frozen scoring and resumable experiment jobs

This workflow packages existing joint models and runs independent computations safely. It adds execution controls, not evidence that market regimes are stable or predictive. Read the [design](execution-design.md), [implementation plan](execution-plan.md), and [preceding robustness findings](joint-validation-results.md).

## Export and score

Use the same numerical sources and dependency versions recorded by the original study:

```sh
regimes freeze-joint \
  --development artifacts/joint-market-9277c7c4c48af92b2d83/development
```

The command prints a content-addressed bundle directory. It copies the original observed-medoid model and freezes its ordered assets, window length, training/calibration dates, data conventions and novelty threshold. Bundles contain market observations inside medoids; keep them local.

```sh
regimes score-joint \
  --bundle artifacts/bundles/joint-bundle-605cdb225dc25b93457706f9 \
  --config configs/frozen_score.yaml
```

The supplied configuration reproduces development validation using the original snapshots. It checks hashes and acquisition sidecars. `start` and `end` select window endpoint sessions; provide at least 63 prior return observations as context. The scorer uses full valid windows, removes raw-price intervals overlapping training, and rejects windows unavailable at `as_of`. Asset order and provider/price conventions must match. There is no fitting, rescaling, filling or automatic source substitution.

`as_of` is a timezone-aware observation-availability cutoff. It does not turn revised adjusted prices into vintage observations. Scores within the bundle's calibration period carry `calibration_reuse=true`; their novelty threshold used later information. Even post-calibration scores remain descriptive, not trading signals.

Outputs include local `scores.parquet`, aggregate `scoring.json`, execution/memory metadata and checksums. The command prints a batch summary path that points to the completed attempt.

## Execute or resume a batch

```sh
regimes run-jobs --config configs/execution.yaml \
  --workers 2 --blas-threads 1 --timeout 3600
```

Run the same command again to reuse every verified completed job. A failed/interrupted job gets a new attempt while previous attempts remain inspectable. Resumption is at job boundaries; a partially completed model fit restarts.

Supported strict job types:

| Type | Inputs | Output |
| --- | --- | --- |
| `score` | Frozen bundle plus snapshot/scoring configuration | Labels, distances, novelty flags and aggregates |
| `refit` | Sealed development, market configuration, training cutoff, seed and candidate budget | Validation-anchor agreement and occupancy |
| `benchmark` | Synthetic window dimensions, seed and optimizer budgets | Fit/score timings, occupancy and execution RSS |

Refit jobs preserve selected K/projections and never evaluate assessment. The earlier-cutoff jobs are retrospective checks because K was selected using later development data. A changed seed, input, numerical dependency, computational source or BLAS thread count gets a different job identity. Changing a display name, worker count or timeout reuses the same computational result. Docs and standalone reporting files are excluded from identities. Relative dataset paths are resolved from the launch working directory, which is recorded and preserved in workers.

Each job uses a fresh Python process. `workers` and `blas-threads` each accept 1–8; actual supported numerical-library thread counts are recorded and checked. Keep their product within available compute capacity. Memory limits are not enforced; select a worker count appropriate for the measured per-worker footprint.

## Failure and integrity behavior

The supervisor logs each worker and exits with failure if any job fails. A timeout terminates that worker's process group; rerun to retry. Ctrl-C stops this supervisor's active workers. Advisory locks serialize duplicate jobs across coordinators and release when a process dies. Completed outputs are sealed before an atomic success pointer is written.

A corrupt completed cache fails verification and is not silently overwritten. Restore its files from a trusted copy, or run against a separate output root to create fresh evidence. Inspect `runs/<id>/summary.json`, worker logs and `<job-id>/attempts/<attempt-id>/status.json`. Incomplete attempts never qualify as a cache hit.

This executor supports Linux and macOS on local filesystems. It does not claim distributed locking, machine-wide scheduling or numerical checkpoints.

## What the memory number means

`peak_rss_bytes` is the OS process-lifetime maximum resident set size through task completion. It includes interpreter/library imports and input arrays. The recorded pre-task high-water value is context, not a baseline to subtract: neither value is an isolated allocation measurement. Linux's KiB and macOS's byte units are normalized explicitly. Per-worker peaks cannot be summed to infer simultaneous system-wide usage. See [Python resource usage](https://docs.python.org/3/library/resource.html).

The [measured execution results](execution-results.md) report all benchmark sizes/seeds, cached resumption and saved-model equivalence. The base numerical package remains NumPy-only; these commands require the existing research extras.
