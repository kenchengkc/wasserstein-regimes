# Research roadmap: from marginal regimes to joint structure

Reviewed September 23, 2026. The original design remains the research contract; this page updates priorities using the completed SPY and cross-asset studies and primary academic sources.

## Current evidence and gaps

Exact univariate W1/W2, chronological evaluation, nine baselines, synthetic controls, saved reports and independent QQQ/TLT/GLD/HYG replication are implemented. The old milestone document's opening description predates those releases. Raw centroid geometry is overwhelmingly scale-driven; shape stability is mixed. TLT and HYG occupy one raw holdout state. These findings motivate testing a different information source—joint dependence—before making economic claims.

The daily provider, strict split machinery and market snapshots are reusable. Joint alignment, multivariate geometry, projection sensitivity, scalable prototype selection and a new empirical protocol are still distinct deliverables. Reusing an exposed holdout to select the next model would not constitute a new confirmatory test.

## Academic projects and decisions

| Primary source / project | Relevant contribution | Decision for this project |
| --- | --- | --- |
| [Horvath, Issa & Muguruza, original Wasserstein regime paper](https://arxiv.org/abs/2110.11848) | Distribution clustering with synthetic and empirical comparisons | Preserve exact univariate models and their results as controls. Published success is not evidence of forecasting value here. |
| [Luan & Hamp, sliced Wasserstein regime classification](https://arxiv.org/html/2310.01285v2) (2025 journal article; revised preprint May 2026) | Fixed projection directions, multivariate synthetic regimes, hyperparameter sensitivity and FX application | Add joint-window sliced W2 with saved projections. Evaluate dependence-only changes and sensitivity to projection count. Our medoid implementation is an alternative, not a reproduction of their projected-centroid algorithm. |
| [Zhuang, Chen & Yang, NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/hash/4a1d69d1f64c6b6df105b15984ca527a-Abstract-Conference.html), [author code](https://github.com/Yubo02/Wasserstein-K-means-for-clustering-probability-distributions) | Distinguishes distance-based and barycenter-based formulations in Wasserstein space | Keep observed windows as medoids initially. Their SDP recovery theorem does not apply to our sampled medoid optimizer; do not transfer that guarantee. |
| [Issa & Horvath, pathwise regime methods](https://arxiv.org/abs/2306.15835), [author code](https://github.com/issaz/signature-regime-detection) | Signature-kernel MMD detects distribution changes on path space, including path dependence | Add lag-vector and identical-bag/different-order controls before introducing a signature stack. Joint distributions alone still discard within-window order. |
| [Rowland et al., AISTATS 2019](https://proceedings.mlr.press/v89/rowland19a.html) | Orthogonally coupled projection estimators and variance analysis | Benchmark projection seeds and counts first. Orthogonal directions are a later alternative, not an unmeasured default improvement. |
| [POT sliced transport documentation](https://pythonot.github.io/gen_modules/ot.sliced.html), [project](https://github.com/PythonOT/POT) | Reference Monte Carlo sliced distance, explicit projection inputs and preprocessing | Use as an optional independent numerical oracle. Keep the initial runtime NumPy-only; do not add an entropic solver to exact one-dimensional work. |

No third-party implementation is copied. References are methodological comparisons, not endorsements of their dependency security or promises of installation compatibility.

## Options and selected first delivery

1. Add more marginal assets: cheap, but unlikely to resolve dependence blindness; revised data and unequal histories still confound comparisons.
2. Add joint sliced transport with observed prototypes: directly tests the next planned hypothesis and exposes an explicit computation budget. **Selected.**
3. Add risk forecasts now: potentially useful, but requires a separate target, loss, baselines and untouched evaluation period. Defer the claim, while specifying its future gate.

The first delivery is a tested joint-distribution engine plus reproducible synthetic and performance evidence. It does not automatically fit the already examined market holdout, select a trading strategy, or claim that synthetic accuracy transfers to markets.

## Staged implementation and acceptance gates

| Order | Deliverable | Acceptance gate |
| --- | --- | --- |
| 1, this change | Fixed-projection sliced W2; bounded candidate medoids; aligned trailing joint windows; safe saved inference | Exact 1D reduction, row-permutation invariance, projection oracle, frozen scales/directions, deterministic seeds, round-trip predictions, no full N-by-N matrix for fixed candidate budget |
| 2, this change | Dependence-only controls, covariance baseline, projection sweep, measured scaling example | Marginals provably identical in a deterministic control; independent train/test synthetic windows; all seeds/counts reported; memory metric labeled; no selective best-run result |
| 3, next empirical study | Common-history, synchronized SPY/QQQ/TLT/GLD/HYG panel with raw and training-scaled ablations | Explicit session/availability contract; no fills; full interval purging; covariance/correlation, marginal and HMM baselines; projection/window/K choices frozen on development; source sensitivity |
| 4 | Refit, null and perturbation validation | Return-block and projection-seed stability, occupancy including single-state flags, stationary dependence null, rare/gradual transitions, outlier sensitivity, chronological anchor sets |
| 5 | Frozen-model scoring and experiment scale | Atomic resumable jobs, isolated fold/seed workers, bounded worker count and BLAS threads, job identity independent of report-only files, validation against frozen schema, measured process RSS |
| 6 | Risk forecasting | Prespecified horizon and proper loss; expanding-window historical, EWMA/GARCH or covariance shrinkage baselines; common information sets; dependence-aware uncertainty; genuinely untouched period or explicit exploratory label |
| 7 | Path-aware or entropic extensions | Prove the blind spot with controls first; compare lag vectors/signatures or debiased Sinkhorn on cost, stability and benefit. GPU work follows profiling. |

## Data expansion

Start the joint empirical study with the existing immutable ETF snapshots on a common date range beginning with the latest inception. Require identical return intervals, session calendars, adjusted-return convention and asset ordering. Never forward-fill a missing return or silently turn a multi-day return into a daily vector. Preserve each source hash and the latest component availability time.

For the next sources, evaluate [FRED/ALFRED](https://fred.stlouisfed.org/docs/api/fred/) for macro series with revision-aware retrieval, [ECB reference-rate downloads](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html) for a documented FX panel, and an entitled [WRDS/CRSP](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/center-for-research-in-security-prices-crsp/) snapshot for survivorship and corporate-action work. These are separate adapters with different calendars and availability assumptions; daily FX fixing rates are not synchronous ETF closing returns. Obtain and record applicable data rights before distributing observations. No paid data is needed for the synthetic engine.

## Scientific stopping rules

Do not treat a forced partition as proof of regimes. Report absent/tiny states and all unsuccessful comparisons. A correlation-switch success is also achievable by a covariance baseline; distributional advantage requires harder controls beyond covariance. Sliced W2 is a different metric from full multivariate W2; finite projections add approximation error relative to sliced distance, not a convergence guarantee to full W2. A large-scale fit is an engineering result, not evidence of economic value.
