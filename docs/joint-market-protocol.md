# Frozen exploratory joint market protocol

Recorded September 23, 2026 before running the new joint models. Configuration: configs/joint_market.yaml.

## Question and interpretation

Determine whether synchronized empirical distributions of SPY, QQQ, TLT, GLD and HYG give useful descriptive structure beyond marginal, covariance and correlation representations. This is a common-history exploratory study. The 2024–August 2026 period has already been examined in earlier marginal work; it is explicitly **not an untouched confirmatory holdout**. No economic or forecasting claim, return target, portfolio or trade optimization.

Use the existing hashed, revised adjusted-close snapshots. Crop each series to the common date range, preserving the price preceding each return. Require exact agreement of dates and return intervals after cropping; reject mismatches instead of joining or filling. Store source hashes, acquisition metadata, common start/end and per-source quality. All assets must be present. One provider is available; independent provider/source sensitivity remains untested.

## Splits and models

Training ends December 31, 2020; validation ends December 31, 2023; assessment ends August 31, 2026. Use 63-return windows, fit stride 5 and score stride 1. Purge windows crossing raw-price boundaries, including the price before the first return. Record maximum input availability. All six models use the same evaluation endpoints.

Select K from 2,3,4,5 on the training-scaled joint model using validation silhouette on a fixed seeded sample of at most 160 validation windows and its true finite-slice distances. No inference of significance from overlapping windows or silhouette. Invalid single-state candidates have no score; fail explicitly if all are invalid. Choose highest score, breaking ties toward smaller K.

Freeze the selected K for raw joint, marginal quantile k-means, covariance k-means, correlation k-means and a full-covariance multivariate Gaussian HMM. The two joint fits use 64 projections, 128 candidate windows and five candidate restarts. Asset scales are population standard deviations of unique training returns, never estimates from validation/test or duplicated windows. Raw joint uses scales one; neither uses within-window normalization.

Marginal features retain all sorted observations for every asset and use the same asset scales, with no quantile-coordinate standardization. Covariance/correlation features use the upper triangle (exclude unit correlation diagonal) and training-only feature standardization. Both feature k-means baselines use 20 starts. HMM uses unique daily vector returns, training-only location/scale, full covariance, three starts and at most 200 EM iterations; report convergence and failures. Before comparing likelihoods or filtering, floor covariance eigenvalues at max(1e-6, largest eigenvalue × 1e-8) in standardized coordinates and record adjustments. Score through causal forward filtering, including known intervening returns, then select common endpoints. Never use smoothed/Viterbi states.

## Diagnostics and sensitivity

Publish validation and assessment occupancy, state counts, switching summaries, agreement with the scaled joint model and contemporaneous mean asset volatility/average correlation by state. These descriptions reuse clustering inputs and do not measure prediction. Include prototype distance novelty using validation 99th-percentile nearest distance, calibrated once; descriptive exceedances are not p-values.

On validation only, use fixed K, scales and optimizer seed with independently supplied projection directions for counts 16/64/128 and seeds 17/42/83. Report every ARI and occupancy; no best setting is selected from this sweep.

Refit the scaled joint model ten times on moving blocks of 26 consecutive training fit windows, preserving each window's internal vector observations; score the fixed validation anchors with frozen original scales and projections. This is **conditional window-block resampling**, not a return-block bootstrap of the full preprocessing pipeline or a confidence interval. Flag both-single-state agreement. It does not complete the later stationary-null, rare-transition, chronological-refit or cross-provider studies in the roadmap.

## Execution, artifacts and acceptance

The development command fits/selects models and seals arrays, configuration, diagnostics and provenance. The explicit assessment command requires that artifact, validates its configuration, acquisition metadata identity and hashes, loads models, and only predicts. Cached assessment additionally binds the exact development checksum digest. Default execution stops after development. Dataset changes fail; published data remain aggregate only. Save per-window assignments locally for audits.

Implementation: panel_baselines.py owns feature geometry and causal multivariate HMM; joint_market.py owns validated configuration, panel preparation, chronological execution and aggregate reports. CLI adds joint-study. Tests must cover source hash/schema rejection, calendar/interval agreement, training-only scales, split input disjointness, causal HMM prefix invariance, archive reload, assessment without development refusal and no fit during assessment. Run actual stages only after these checks and code review. Save aggregate results and measured conclusions, including adverse findings.
