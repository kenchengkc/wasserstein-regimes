# Implementation plan and acceptance criteria

## Delivery strategy

Build a local, reproducible research tool before adding scheduled services or joint-distribution machinery. The critical path is data conventions -> exact distribution geometry -> chronological validation -> evidence of value. The plan is complete enough to execute; completing it does not imply the target system already exists.

Current repository deliverables: documentation, GPL-3.0-only license, Python packaging, exact equal-size W1/W2 reference functions and clustering, rolling windows, synthetic smoke example, and numerical tests with CI. The milestones below describe the remaining production research implementation. The reference code is a correctness seed, not the whole M1–M7 release.

Effort estimates assume one experienced quantitative Python developer, existing data entitlement where needed, and part-time domain review. They are planning estimates, not commitments. First daily research release: approximately 4–6 weeks. The hourly reproduction and advanced extensions add approximately 2–4 weeks depending on data cleanup.

## Milestones

| ID | Work and dependencies | Deliverables | Acceptance gate | Estimate |
| --- | --- | --- | --- | --- |
| M0 | Finalize research contract | Signed-off metric/objective conventions, daily and paper profiles, split policy, source choice | Every unspecified paper setting appears in a discrepancy log; target data range and price basis are explicit | 1–2 days |
| M1 | Harden mathematical core; after M0 | Immutable window batches; chunked W1/W2 estimator; restarts, diagnostics, safe serialization | Numerical oracle agreement, monotone objective except documented reset events, prediction/serialization invariance, invalid-input failures | 3–4 days |
| M2 | Data acquisition; after M0, can overlap M1 work | CSV plus one daily provider adapter, canonical Parquet, action/calendar validation, snapshot manifests | Same snapshot/config produces identical returns/windows; gaps and actions handled without artificial spikes or invented observations | 3–5 days |
| M3 | Synthetic suite and baselines; after M1 | GBM, Merton, equal-moment shape controls, null regime controls; volatility/moment/GMM/HMM baselines | Label-invariant metrics, matched sample/timing contracts, all random seeds saved, 50-path benchmark script | 3–4 days |
| M4 | Chronological experiment runner; after M1–M3 | Nested forward splits, interval purging, operational replay, MMD/uncertainty/stability, model selection | Future-data perturbations cannot change prior predictions; no shared inputs in strict splits; HMM uses causal filtering | 4–6 days |
| M5 | Reports and first daily study; after M4 | CLI, manifest-driven HTML report, centroid/timeline/MMD/stability figures, held-out SPY comparison | Clean-machine rerun works; report states all exclusions, timing assumptions, uncertainty and unsuccessful comparisons | 3–4 days |
| M6 | Hourly paper-period reproduction; after M2–M5 | Licensed hourly snapshot, W1 paper profile, GBM/Merton tables and comparable plots, discrepancy report | Vendor/session/overnight policies explicit; retrospective versus causal results separate; no unsupported replication claim | 3–5 days plus data access |
| M7 | Research release and operational inference; after M5 | Package release, frozen-model batch scoring, atomic artifacts, data-health and novelty outputs | Offline deterministic tests pass; bounded-memory benchmark; stale/missing data returns explicit unavailable status | 2–3 days |
| M8 | Joint and path-aware extensions; after M7 and evidence of need | Sliced-Wasserstein k-medoids, correlated-vector and lag-vector controls, projection sensitivity | Detects a dependence-only change beyond marginal baselines; projection/scale state saved; approximation identified | 5–8 days |

No need to wait for paid hourly data to execute M1–M5. Use synthetic and appropriately sourced daily data to build the entire evaluation path. Do not purchase data automatically as part of an adapter run.

## Concrete engineering backlog

### M1: Numerical package

- Define `metric` separately from objective exponent, distance outputs, and stopping metrics. Support W1 and W2 only initially.
- Extract reference functions into tested transport and model modules. Add chunked distance calculation, deterministic empty-cluster handling, 20 restarts, configurable tolerances, and retained convergence histories.
- Define `WindowSpec(length,stride)` and validate positive integers. Preserve raw price intervals and endpoint timestamps. Keep exact atoms, including duplicates and extremes.
- Add model JSON/NPZ round trips, schema version checking, and safe loading. Expose scikit-learn-style `fit`, `predict`, `transform`, and `fit_predict` semantics without promising compatibility before estimator checks pass.

### M2: Data foundation

- Build a local CSV adapter first, then an optional daily adapter. Follow the [source recipes](data.md).
- Implement typed schemas and validators with synthetic split/dividend, missing-bar, duplicate, timezone, DST, and early-close fixtures.
- Build immutable raw snapshots and canonical returns. Define `available_at` and assumed vendor latency before writing replay code.
- Add one acquisition CLI command with explicit provider, symbol, date range, frequency, adjustment policy, and destination. Implement bounded retries and atomic output.

### M3–M4: Research validity

- Reproduce paper raw-moment features and add interpretable mean/std/skew/excess-kurtosis features as a separately named baseline. Use training-only transforms.
- Fit a Gaussian HMM with multiple starts and reasonable covariance floors; implement filtering for operational evaluation. Keep retrospective decoding separately named.
- Build nested chronological folds based on raw observation intervals. Include a strict split and an operational replay mode, each labeled in outputs.
- Compute all algorithms' primary MMD diagnostics in the original return domain. Freeze kernel bandwidths from historical data.
- Build seed and refit stability measurements using adjusted Rand index and centroid matching. Compare models on a common anchor set, not on changing samples; align refits by minimum-cost centroid matching and flag poor matches rather than forcing semantic continuity.
- Add optional novelty thresholds and hysteresis only after the raw clustering results are stable. Calibrate them within historical validation blocks.

### M5–M7: User workflow

Target CLI commands (planned, not available in the reference package):

```text
regimes fetch --config configs/data_spy_daily.yaml
regimes validate --snapshot <snapshot-id>
regimes run --config configs/daily_walk_forward.yaml
regimes reproduce --config configs/paper_w1.yaml
regimes report --run <run-id>
regimes score --model <model-id> --snapshot <new-snapshot-id>
```

Configuration must include source, data cutoff, frequency, return/adjustment policy, window and stride, metric, k, fit schedule, split policy, hyperparameter search, seeds, baselines, evaluation-pair budget, bandwidth rule, bootstrap settings, and artifact destination. Validate unknown keys rather than ignoring typos.

Reports should be self-contained and generated only from saved artifacts. Include a compact model card: intended scope, input period, preprocessing, validation results, known failure modes, temporal guarantees, data rights, and whether deployment is supported. Scheduled scoring should refuse an incompatible frequency/window/preprocessing model and mark stale data explicitly. Model retraining must create a new version; no hidden online updates.

## Experiment matrix

| Experiment | Purpose | Fixed controls | Success interpretation |
| --- | --- | --- | --- |
| Dirac and equal-size empirical measures | Mathematical correctness | Hand-calculated W1/W2 and barycenters | Exact or floating-point agreement, no research claim |
| Paper GBM | Gaussian variance/drift regime benchmark | Paper parameters, 50 paths, recorded switching intervals | Balanced accuracy, regime recall, centroid error and delay with uncertainty |
| Paper Merton | Jump-sensitive benchmark | Same seeds/paths for each model, explicit uncompensated drift | Determine whether improvement persists across paths; accept failure to replicate |
| Matched mean/variance shape change | Isolate distribution information | Normal vs scaled Student-t with finite variance, or symmetric matched-variance mixture; control finite-sample variation | W1/W2 compared to mean/std baseline; not guaranteed to beat a four-moment model |
| Matched first four moments | Stronger shape test | Construct and verify distinct discrete distributions with equal first four moments | Test full-distribution benefit against the exact four-moment baseline |
| No-regime stationary series | Detect overinterpretation | Same family and distribution throughout | Report artificial segmentation, instability, novelty false alarms; k-means always partitions if k>1 |
| Short/gradual transitions and rare regimes | Delay and imbalance stress | Transition lengths below and above w; unequal state occupancy | Show tradeoff rather than choosing only easy long episodes |
| Contaminated series | Robustness and data quality | Isolated bad ticks versus genuine jumps; W1/W2 side by side | Measure sensitivity without deleting true tails |
| Daily SPY held-out study | Practical descriptive validity | Common snapshots, splits, tuning budget and bandwidth protocol | Stable and interpretable held-out structure, not claims of true regime labels |
| Cross-provider and cross-asset | Robustness of result | Matched dates/conventions; separate asset models | Quantify assignment disagreements and source sensitivity |
| Marginally identical correlation switch | Justify multivariate extension | Same per-asset distributions, changed joint correlation | Marginal method should be blind; joint method must be evaluated independently |
| Identical bags with changed order | Show temporal blind spot | Exactly same atoms under permutation | 1D distance must be zero; lag-vector extension may distinguish |

For a deterministic first-four-moment control, use support `x=0,1,2,3,4,5`, base weights `1/6`, and perturbations `±0.01*[1,-5,10,-10,5,-1]`. Both weight vectors are positive, normalized, and have identical moments of orders 1–4 because the fifth finite difference annihilates lower-degree polynomials. Translate/scale both supports to return units. With 300 atoms, each weight times 300 is an integer, so exact equal-size empirical test windows can demonstrate the distinction without sampling noise. Add stochastic samples as a separate statistical-power experiment.

## Testing strategy

Numerical tests should compare W1 to `scipy.stats.wasserstein_distance`, W2 squared to an independent small optimal-assignment or transport calculation, and the W2 embedding to a standard k-means backend under matched initialization. SciPy's `wasserstein_distance` is W1, not W2. [SciPy reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html).

Test permutation invariance, translation/scaling behavior, zero-distance duplicates, medians versus means on skewed samples, sorted centroid validity, deterministic restarts, one cluster, invalid k, mismatched atom counts, nonfinite inputs, and early/max-iteration exits. Weighted/unequal-length support needs separate exact integration tests when added.

Temporal tests are release-blocking: change all observations after cutoff `t` and assert predictions and fitted artifacts through `t` remain unchanged; assert each signal's availability is no earlier than every input's availability; check strict splits share no input prices; verify no post-cutoff normalizer or bandwidth state is read. Test label postprocessing online and in one-shot replay for equivalence.

Data tests cover actions, timezone changes, nontrading days, early closes, missing bars, duplicates, source revisions, terminal returns, and API error payloads. Integration tests run a fully synthetic snapshot through returns -> windows -> fit -> score -> saved report. Provider tests are optional and must not become a condition for offline mathematical CI.

Performance targets for the research implementation: on a recorded modern laptop CPU, fit 10,000 windows of 126 atoms, k<=6, 20 restarts within two minutes and under 1 GB peak process memory; score 1,000 windows within two seconds, excluding network I/O. These are proposed gates to measure and revise based on profiling, not benchmarks already achieved. Capture CPU, RAM, BLAS/thread settings, versions, workload, and wall time.

## Model acceptance and stopping rules

Before opening the final holdout, write down occupancy and stability criteria, primary diagnostics, pair-sampling and bootstrap budgets, and a near-best selection tolerance. Do not select the model that best colors a remembered crisis. Reserve a no-deployment outcome if results depend on source quirks, one seed, or one bandwidth.

For engineering release, require all numerical, temporal, and data-integrity tests to pass, deterministic artifact replay within documented floating tolerances, and clear unavailable/error states. For a research claim of distributional advantage, require improvement on shape-controlled synthetic tests and stable held-out comparisons with uncertainty; a successful implementation may legitimately find no market-data advantage.

For operational use, additionally require frozen preprocessing, an auditable model version, data freshness checks, calibrated novelty behavior, and a documented retraining schedule. If a trading application is later requested, its execution timing and transaction-cost validation become separate acceptance gates.

## Risks and decisions to revisit

| Risk | Mitigation or explicit limit |
| --- | --- |
| Short windows poorly estimate tails | Window-length sensitivity, uncertainty, no rare-event probability claims from a few observations |
| Volatility dominates distances | Retain primary raw-return geometry; run a separate shape-only normalization ablation |
| Overlap makes validation look stronger | Interval purging, disjoint evaluation pairs, dependence-aware bootstrap, operational/statistical split distinction |
| The market changes outside known regimes | Distance-based novelty, versioned refits, no automatic assumption that nearest means familiar |
| Rare regimes become tiny/outlier clusters | Occupancy and seed stability diagnostics, k sensitivity; do not enforce balance as a universal truth |
| Retrospective revisions change returns | Immutable snapshots, source-version metadata, point-in-time qualification |
| Multivariate scaling/projections alter answers | Training-only scaling, stored projections, projection-count sensitivity, medoids first |
| Data entitlement blocks reproduction | Continue daily/synthetic work; report unavailable hourly evidence without fabricating results |

Default decisions: private repository, daily SPY first, GPL-3.0-only, W1 plus squared W2, exact full atoms, local CPU/Parquet, no order routing, and a separate licensed hourly reproduction. Provider purchase and eventual publication of licensed-data outputs require an actual source choice and its terms; neither is necessary to begin implementation.
