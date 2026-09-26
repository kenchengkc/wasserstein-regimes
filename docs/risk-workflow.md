# Reproduce the risk forecasting study

Read the [frozen protocol](risk-protocol.md) before changing any configuration. The five-asset snapshots, acquisition sidecars and frozen joint bundle are the same inputs used in phase 5. The forecast comparison is exploratory because the assessment prices were previously inspected.

From the repository root, in the original pinned research environment:

```sh
export PYTHONPATH=src
export VECLIB_MAXIMUM_THREADS=1
python -m wasserstein_regimes.cli run-jobs \
  --config configs/risk_jobs.yaml --workers 1 --blas-threads 1 --timeout 3600
```

If the frozen bundle is absent, export it using the [execution workflow](execution-workflow.md). The bundle contract enforces exact numerical sources and dependency versions. The [data guide](data.md) explains acquisition; no raw prices or observed medoids are distributed. A newer provider download can differ and cannot reproduce the original hashes.

The command prints a batch summary path. Its job entry points to a completed attempt containing:

| File | Meaning |
| --- | --- |
| `forecasts.parquet` | Local origin dates, availability, states/support, five forecasts, matured targets and losses; pending tails have no target/loss |
| `risk.json` | Frozen config, bundle digest and aggregate metrics, yearly descriptive slices and paired block-bootstrap intervals |
| `task.json` | Task result used by the executor |
| `execution.json` | Fresh-worker runtime, threads and process high-water RSS |
| `checksums.json` | Complete sealed inventory |

Run the same command again to verify cached resumption. Changes to computational code, config, input bytes or dependencies create another identity. Report-only documentation changes do not. Job failure leaves its attempt inspectable and never publishes a success pointer.

The loader rejects missing/unaligned daily intervals, asset-order or return-convention changes, conflicting sidecars, unavailable forecast origins and nonmonotonic availability. The final five forecasts normally remain pending because their full horizon extends beyond the snapshot. An earlier `as_of` also withholds not-yet-available targets. No future value may enter the state-to-risk map until its horizon matures.

Publish aggregates only: origin/maturity ranges, occupancy counts, support summaries, method mean losses, paired differences and all block sensitivities, input/source hashes and execution metadata. Keep `forecasts.parquet`, bundle/model bytes and raw data local. Record the exact reviewed source commit and inventory digest with any published evidence. Bootstrap intervals describe one frozen selected model/rule, not selection uncertainty or confirmatory significance.
