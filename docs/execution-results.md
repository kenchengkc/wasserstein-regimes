# Frozen scoring and execution measurements

**Frozen scoring reproduces every saved validation label, and the complete 11-job batch resumes from verified artifacts.** This phase delivers execution infrastructure; it does not change the [robustness findings](joint-validation-results.md) or establish predictive value.

The batch contains one saved-model scoring job, four historical training-prefix/seed refits, and six synthetic scaling benchmarks. Every job ran in a distinct fresh process, with at most two workers configured and one numerical-library thread requested per worker. All observed threadpool counts were within that limit. The second invocation used one worker and reused all 11 original attempts without recomputation.

## Saved inference and integrity

The scaled joint bundle preserves the original model bytes, asset order, scales, projections and calibration threshold. Scoring the historical validation snapshots reproduced **690 of 690 dates, labels and availability timestamps exactly**, with occupancy `[0, 287, 403]`. Every novelty score is labeled retrospective calibration reuse because these dates precede the threshold's calibration cutoff. Revised source histories remain explicitly non-vintage.

The scorer required no fitting and peaked at **241.1 MiB process RSS** through task completion. Its worker elapsed time was 1.15 seconds, including task imports and verification but excluding process startup and lock waiting. Observed medoids and per-window scores remain local; public evidence contains only aggregates and fingerprints.

## Scaling benchmark

Each independent job creates float64 Gaussian windows with 63 returns across five assets, then fits K=3 sliced-W2 medoids with 64 projections, 128 candidates and three restarts. This is an engineering load, not a market regime experiment. Fit/score times measure those calls; RSS includes the interpreter, libraries and input arrays. No warm-up pass or repeated timing distribution is claimed.

| Windows | Seed | Input MiB | Fit seconds | Score seconds | Peak process RSS MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 17 | 2.40 | 0.324 | 0.044 | 93.8 |
| 1,000 | 42 | 2.40 | 0.322 | 0.043 | 76.9 |
| 10,000 | 17 | 24.03 | 1.493 | 0.438 | 118.6 |
| 10,000 | 42 | 24.03 | 1.496 | 0.438 | 118.1 |
| 50,000 | 17 | 120.16 | 6.675 | 2.223 | 232.1 |
| 50,000 | 42 | 120.16 | 6.801 | 2.218 | 214.6 |

These are two seeds per size measured on the recorded macOS/arm64 environment while up to two jobs ran concurrently. They are not hardware-independent bounds or evidence of parallel speedup. Peak RSS varies with library and allocator behavior as well as input size. It is a process-lifetime high-water mark through task completion, not incremental fit memory. The earlier Python-traced allocation benchmark measures a different quantity.

Keep concurrency conservative: the four full market refit workers peaked at **275.5–289.5 MiB each**, higher than these synthetic benchmark workers. Per-process peaks do not identify simultaneous machine-wide RSS, and this executor does not impose an OS memory limit.

## Isolated prefix/seed checks

The jobs reuse frozen K/projections and score the same 690 validation anchors. They confirm that job isolation preserves the previously observed sensitivity; they are not new untouched evaluations.

| Training end | Seed | ARI versus reference | Validation counts |
| --- | ---: | ---: | --- |
| 2017-12-31 | 42 | 0.206 | [0, 103, 587] |
| 2017-12-31 | 83 | 0.943 | [0, 395, 295] |
| 2020-12-31 | 42 | 1.000 | [0, 287, 403] |
| 2020-12-31 | 83 | 0.405 | [412, 278, 0] |

The full-history seed-42 row reproduces the reference. Its self-agreement is not independent stability evidence. Earlier-prefix jobs still use K selected with later development information.

## Reproduction and recovery tests

Follow the [execution workflow](execution-workflow.md). The frozen batch is `configs/execution.yaml`; the scoring specification is `configs/frozen_score.yaml`. Execution source is commit `e1237ce`; the recorded computational file inventory and dependency versions bind each job. The original bundle is `joint-bundle-605cdb225dc25b93457706f9`.

Initial batch: `407c91f9ac4f43a4b665b42060a4ba5a`. Cache-only batch: `f47d156eb101465ca212f318b4e62557`. Every attempt inventory and job input/source fingerprint was verified before publication. Display/report changes do not change job identities; numerical source, input, dependency or BLAS-thread changes do.

Automated tests exercise real subprocess isolation, cache reuse with a changed worker count, two coordinators requesting the same job, timeout/retry, interrupted attempts, and live supervisor cancellation with worker cleanup. They also reject corrupt output inventories, changed inputs during computation, symlink/sidecar identity mismatches, mixed-generation bundle exports and incompatible scoring contracts. Independent review identified two provenance defects; regression tests reproduced both before fixes and evidence execution.

[Public aggregate evidence](https://github.com/kenchengkc/wasserstein-regimes/tree/e846df22a91a2c005cad36a4c8fd4bc7c4968424/results/execution) includes all 11 job results, exact computational specifications/fingerprints, runtime threadpool/RSS metadata and checksums. The local job store retains per-attempt status and logs. This supports process interruption recovery on local Linux/macOS filesystems; it is not a power-loss durability guarantee or a distributed scheduler.

## Next research gate

Roadmap phase 5 is delivered. Phase 6 requires a separate frozen risk-forecast target, horizon, proper loss, baselines and dependence-aware uncertainty. The previous market periods remain exposed, and the regime representation remains fragile. This infrastructure makes such experiments reproducible; it does not justify selecting a trading rule or claiming a useful risk forecast.
