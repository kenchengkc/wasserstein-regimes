# Joint regime robustness protocol

Frozen before executing this suite. This is roadmap phase 4, following the exploratory five-ETF panel. The question is whether partitions withstand resampling and contamination, and whether similar apparent structure occurs without regime changes. No predictive or confirmatory market claim is authorized by this study.

## Decisions

Use the existing sliced-W2 medoid engine, unchanged, with covariance clustering as a synthetic comparator. A larger empirical universe would not address forced partitions. Forecasting would require new targets and evaluation data. This phase therefore tests the current model's failure modes before either expansion.

The existing development artifact supplies the frozen K=3 reference, directions, scales, configuration and data identities. Verify its checksum inventory, numerical source files, dependency versions and acquisition metadata. New orchestration and CLI code may differ; existing numerical source files may not. Do not read assessment assignments or evaluate observations after 2023. Original adjusted-price snapshots remain local and unchanged. Public artifacts contain aggregate diagnostics and provenance only.

## Market diagnostics

All comparisons score the same 690 purged validation windows (2021-04-06 through 2023-12-29). Report counts including empty states, single-state flags, ARI against the saved reference, reference occupancy, convergence and mean squared distance to the nearest medoid. ARI is agreement, not accuracy. Nearest-medoid costs use each refit's own scales, so cost levels are not directly comparable across changed scales.

- **Return resampling:** stationary bootstrap of synchronized daily training vectors, sharing indices across assets, with circular continuation and geometric block length. Expected block lengths 63, 126 and 252 returns; ten deterministic resamples per length. Recompute asset scales from the resampled daily vectors, rebuild trailing 63-return windows, then fit at stride 5. Keep K=3, 64 reference directions, 128 candidates and five restarts fixed. Joined blocks create synthetic boundaries; report how many fit windows cross a restart, including circular wrap. This conditions on previously selected K and projections; it is not full selection uncertainty or a confidence interval.
- **Chronological refits:** expanding training history ending in 2015, 2017 and 2020, with unique-return scales recomputed per prefix. All score the same later validation anchor set. K was selected using the full development history, so earlier refits are retrospective sensitivity checks, not historical deployment simulations.
- **Candidate budget:** 64, 128 and 256 candidates crossed with optimizer seeds 17, 42 and 83, with reference scales/directions. Publish every row; the 128/42 self-comparison is explicitly flagged and excluded from sensitivity summaries.
- **Training outliers:** contaminate a seeded 0.1% or 1% of daily training vectors by adding an independent signed 10-training-standard-deviation shock to every asset. Recompute scales/windows and refit; evaluate clean validation anchors. Save counts and perturbed scales, not raw vectors. No clipping or robustness fix is selected after results.

## Synthetic controls

Use five assets, 3,000 daily vectors, L=63, fit stride 5, 64 seed-42 projections, five restarts and seeds 17, 42, 83, 101, 211. Disjoint raw-return segments: training [0,1500), calibration [1500,2250), test [2250,3000); include only windows fully contained in each segment. Fit scales on training vectors. Fit and evaluate joint medoids and covariance KMeans on identical windows.

**Stationary dependence nulls:** constant equicorrelation 0.45, AR(1) coefficient 0.25, 512 burn-in vectors; Gaussian or multivariate Student-t(5) innovations with a shared radial divisor, variance normalized to one. Forced K=3 and 128 candidates. Report test occupancy, switching/dwell, silhouette on a seeded sample of up to 160 test windows, and exceedance of each model's calibration 99th-percentile nearest-center distance. These ten Monte Carlo runs describe artifacts of partitioning a single stationary law. They are not a calibrated hypothesis test or a universal false-regime rate; Student-t burn-in is an approximation to stationarity.

**Rare dependence changes:** independent Gaussian daily vectors, rho=0.2 except 126-return rho=0.8 episodes starting at 950, 1800 and 2550. The last episode occupies 16.8% of test daily vectors, with only 64 pure high-state windows. Fit K=2; candidate budgets 64/128/256 crossed with all five seeds.

**Gradual dependence changes:** in each raw-return segment, rho rises from 0.2 to 0.8 over a centered 250-return linear ramp of intermediate correlations strictly between the two endpoints. Fit K=2 with the same candidate/seed grid. Marginal population laws stay N(0,1). Windows touching a ramp or containing two regimes are excluded from pure-state recovery metrics; report their number separately. No binary truth is assigned to the ramp.

For changing controls report pure-test ARI, per-state recall and balanced accuracy using label mapping learned on pure training windows only. Include full test occupancy. For rare episodes report the first five consecutive mapped high predictions within the test episode; failure is right-censored at the episode end, and observed delay includes the rolling-window lag. Do not report only successful delays. Covariance is expected to recognize correlation changes; this control does not establish a distributional advantage beyond covariance.

## Artifacts and acceptance

Strict YAML schema and deterministic seeds. One command produces a sealed directory with config, manifest, all result rows, and HTML; repeated execution verifies and reuses that directory. Identity binds source hashes, dependency versions, snapshots and parent development checksum digest. Atomic rename exposes only complete runs. No parallel workers in this phase.

Tests must cover synchronized resampling, deterministic generation, constant population parameters under nulls, disjoint raw-return splits, training-only scaling and label mapping, contamination immutability, censored delays, frozen parent compatibility and cache tamper detection. Scientific success is a complete truthful report, including poor outcomes. Do not optimize the protocol after seeing results.

## Sources

[Politis & Romano, The Stationary Bootstrap](https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870) motivates geometric-length resampling. Its asymptotic theory is not claimed for the selected clustering pipeline. [Luan & Hamp](https://arxiv.org/html/2310.01285v2) motivate multivariate synthetic controls and hyperparameter sensitivity; our observed-medoid optimizer remains a different algorithm.
