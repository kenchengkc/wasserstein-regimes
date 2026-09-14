# System design

## 1. Objective and scope

Build a reproducible Python research package that discovers recurring market conditions from empirical return distributions, assigns newly completed windows to frozen centroids, and explains differences through quantile curves and distribution diagnostics.

The first release handles one asset's distribution through time. The reference implementation in this repository is deliberately smaller than this target design; the status and remaining milestones are in [the implementation plan](implementation-plan.md).

Three modes have different temporal contracts:

| Mode | Fit information | Output interpretation |
| --- | --- | --- |
| Retrospective | Entire chosen sample | Historical descriptive segmentation, including overlap-vote plots |
| Walk-forward | Only information available before a prediction block | Causal assignment using frozen transformations and centroids |
| Scheduled inference | Saved approved model and newly completed windows | Timestamped assignments, novelty scores, and data-health indicators |

Clustering does not impose contiguous states or a transition model. The same regime can recur, and adjacent windows may switch labels. Start with raw assignments; any hysteresis or persistence rule is an explicit, separately evaluated causal postprocessor. An order-execution service is outside the first release.

## 2. Mathematical specification

For adjusted daily prices `P_t > 0`, define `r_t = log(P_t) - log(P_(t-1))`. If a provider supplies simple total returns directly, use `log1p(R_t)` instead and reject `R_t <= -1` with an explicit terminal-event policy. Do not apply log differences to a return series.

A completed window of `w` returns ending at `t` defines

$$\mu_t=\frac{1}{w}\sum_{j=0}^{w-1}\delta_{r_{t-j}}.$$

Let `q_t` be those returns sorted ascending. It is the exact empirical quantile representation on `w` equal-mass intervals. Use every atom by default; histogram bins, kernel density estimation, and quantile compression are unnecessary in the initial implementation.

For equal-length vectors:

$$W_p(\mu_i,\mu_j)^p=\frac{1}{w}\sum_{a=1}^w|q_{i,a}-q_{j,a}|^p.$$

The clustering objective is

$$J_p=\sum_{i=1}^M \omega_i W_p(\mu_i,\nu_{z_i})^p,\qquad p\in\{1,2\}.$$

Default `omega_i=1`; observation weights are a later extension. Distinguish observation weights across windows from atom weights within each distribution.

For W1, update each centroid atom with a median across member windows. For squared W2, use a mean. Coordinate medians and means of nondecreasing vectors remain nondecreasing, so the centroid is a valid equal-mass distribution. A centroid is a Wasserstein barycenter, not the pooled mixture of member returns; pooling changes the optimization problem.

For W2, `q/sqrt(w)` is an exact Euclidean embedding. A standard k-means backend can be used without approximation, provided it receives all order statistics and does not standardize individual quantile coordinates. Its initialization, stopping, and objective scaling must match the documented contract. This equivalence is a useful correctness oracle for an independent implementation.

For unequal atom counts or weighted empirical measures, use exact integration over the merged cumulative-mass breakpoints in one dimension. Do not pad, truncate, or pair arbitrary quantile indices and call the answer exact. Defer this feature until a weighted implementation is tested. A fixed midpoint grid is an optional approximation with a reported resolution and distance error.

### Optimization behavior

The production implementation should sort each input once, initialize distinct distributions, alternate assignment and centroid updates, and track the powered objective after every iteration. Use 20 seeded restarts initially; keep the lowest objective among converged runs. Standard k-means++ is valid for the W2 embedding. For W1, distance-proportional seeding can be a heuristic, with no transferred k-means++ guarantee.

Require `k <= number of distinct distributions`. On an empty cluster, reseed using the highest-loss eligible observation with a stable tie-break, recompute assignments, and log the event. Recompute final labels against returned centroids. Mark iteration-limit exits as unconverged. Persist objective history, actual iterations, seed, restart count, and empty-cluster events. The reference implementation uses simpler seeded distinct-observation initialization and exposes convergence status.

Stop on unchanged assignments or sufficiently small objective change plus a small centroid displacement; keep a hard iteration cap. Tolerances must distinguish return units from squared-return units. Floating-point ties use a deterministic label order. Global optimality is not guaranteed.

### Interpreting centroids

Store arbitrary cluster IDs, then derive descriptive names from training data: low dispersion, negative skew, heavy downside tail, or elevated dispersion. A high-volatility cluster is not automatically a bear market. Report centroid quantiles and distributions of member-window mean, standard deviation, skewness, and empirical tail loss. Centroid tail metrics are descriptive; a 35- or 63-observation window cannot estimate very rare tail probabilities precisely.

## 3. Architecture and dependencies

Use one installable Python package with a `src/` layout. Begin with Python 3.11+, NumPy, SciPy, pandas, PyArrow, scikit-learn, matplotlib, and a small CLI. Keep optional `hmmlearn`, provider SDKs, and Python Optimal Transport (POT) in extras. Use local Parquet plus JSON and NPZ artifacts; add DuckDB when query workloads justify it. No database server, GPU, distributed scheduler, or web service is needed for the first release.

The committed reference core needs only NumPy at runtime. Dependency bounds in its package metadata are compatibility ranges, not a reproducibility lock. Before the research release, generate and commit a resolved lock for each supported Python/platform target, and persist the actual installed versions in every run.

```mermaid
flowchart LR
    A[Provider adapters] --> B[Immutable raw snapshots]
    B --> C[Validation and corporate actions]
    C --> D[Canonical returns]
    D --> E[Trailing empirical distributions]
    E --> F[Chronological experiment runner]
    F --> G[W1 or squared W2 model]
    F --> H[Matched baselines]
    G --> I[Held-out diagnostics]
    H --> I
    I --> J[Versioned report and model artifacts]
    J --> K[Frozen-model inference]
```

Target module layout (items beyond the current `core.py` are planned):

```text
src/wasserstein_regimes/
  core.py                      # existing mathematical reference
  config.py                    # validated immutable experiment configuration
  data/
    base.py                    # provider contract and capabilities
    csv.py                     # local licensed/user-supplied files
    yahoo.py                   # optional exploratory adapter
    alpha_vantage.py            # optional licensed API adapter
    french.py                  # research return files
    schema.py                  # bars, actions, returns, provenance
    validation.py              # calendar, duplicates, gaps, adjustments
    storage.py                 # immutable snapshots and content hashes
  distributions/
    windows.py                 # endpoints, overlap, observation intervals
    empirical.py               # full sorted atoms and metadata
  transport/
    one_dimensional.py         # exact W1/W2 distances and barycenters
    sliced.py                  # later multivariate projected distances
  models/
    wasserstein.py             # fit, predict, transform, serialize
    baselines.py               # moments, volatility, GMM, filtered HMM
    postprocess.py             # optional causal hysteresis
  evaluation/
    splits.py                  # interval-aware purging and outer folds
    metrics.py                 # MMD, stability, occupancy, novelty
    synthetic.py               # GBM, Merton, shape and dependence controls
    bootstrap.py               # dependence-aware uncertainty
  experiments/
    runner.py                  # config -> reproducible artifact directory
    manifest.py                # source/code/dependency/config identity
  reporting/
    plots.py                   # quantiles, assignments, confusion, timing
    report.py                  # self-contained HTML and static exports
  cli.py
configs/                       # planned paper and walk-forward profiles
tests/                         # numerical, temporal, data, integration tests
examples/                      # runnable examples; notebooks only as clients
```

Numerical modules must not access the network or read credentials. Adapters must not fit models. Reports consume saved artifacts and must not retrain implicitly. The CLI calls package functions; notebooks never own essential preprocessing logic.

## 4. Interfaces and artifacts

Proposed public APIs, not yet implemented beyond the reference core:

```python
provider.fetch(request: DataRequest) -> RawSnapshot
validate(snapshot: RawSnapshot, policy: DataPolicy) -> ValidationResult
build_returns(bars: BarTable, policy: ReturnPolicy) -> ReturnTable
build_windows(returns: ReturnTable, spec: WindowSpec) -> WindowBatch

model = WassersteinKMeans(metric="w2", n_clusters=3, n_init=20, seed=42)
model.fit(train_windows)
labels = model.predict(test_windows)
distances = model.transform(test_windows)  # true Wp, shape M x k
run_walk_forward(config: ExperimentConfig) -> RunManifest
```

`WindowBatch` contains `atoms[M,w]` float64, start/end indices, asset ID, return-window end time, input price interval, availability time, source snapshot ID, and quality flags. Batch constructors validate sortedness, finiteness, equal sample size, and metadata alignment. Arrays should be immutable at public boundaries.

`ModelArtifact` contains schema version, metric, objective exponent, sorted centroid atoms, training time bounds, preprocessing state, window spec, seed, hyperparameters, objective history, convergence status, dependency versions, and source/config/code hashes. Prefer JSON plus NPZ loaded with `allow_pickle=False`; validate dimensions and checksums on load. Do not rely on opaque pickle files for long-lived interchange.

`PredictionTable` contains window ID, model version, `as_of`, `available_at`, assigned ID, distances to every centroid, nearest/second-nearest distance margin, novelty flag, quality flags, and optional postprocessed state. A margin is a geometric diagnostic, not a calibrated probability. Novelty is the distance from known centroids relative to a threshold calibrated on historical validation data; nearest-centroid assignment alone cannot identify a new regime.

Each run writes:

```text
artifacts/<run_id>/
  config.json
  manifest.json
  validation.json
  model.json
  centroids.npz
  predictions.parquet
  metrics.json
  report.html
  figures/
  logs.jsonl
```

Use content hashes plus an execution timestamp to identify runs. The manifest records retrieval time, source file hashes, provider revision, price adjustment policy, calendar version, units, split boundaries, evaluation samples, git commit, dirty-worktree diff hash if applicable, seeds, and runtime environment. Exclude API keys and authorization-bearing URLs. Publish only artifacts permitted by the data license; small centroids can still reveal licensed data.

## 5. Temporal validity and model selection

All windows are trailing. A window ending at close `t` becomes available only after that close and the provider's publication delay. It may drive the next executable decision, never a trade earlier within its own window.

For model selection, use chronological outer train/test blocks and inner train/validation splits. Fit scaling, kernel bandwidths, novelty thresholds, cluster naming, and any trading rule within the corresponding historical training/validation period. Never tune `k`, `w`, or preprocessing against the final test timeline.

Implement two explicit split policies:

1. **Strict nonoverlap evaluation (primary statistical comparison):** purge windows whose raw input intervals intersect across partitions. Include the price preceding the first return; a `w`-return window uses `w+1` prices. Select boundaries from actual interval metadata rather than a guessed row count. Add a documented separation gap for remaining temporal dependence and test sensitivity to its length.
2. **Operational replay:** new test windows may contain pre-cutoff historical returns, since those would be known in production. The model still uses only past fitting data. Report this as operationally causal but statistically dependent evaluation, separately from strict nonoverlap scores.

No random cross-validation over overlapping windows. A forward-only fold needs purging around the next block, not arbitrary training samples from its future. Overlapping windows increase the number of rows, not the number of independent observations.

Initial daily protocol: development through 2018, rolling outer evaluation blocks from 2019 onward, and reserve 2024-01-01 through 2026-08-31 as a final untouched report period. These are proposed research boundaries, not data already acquired. The final period is a retrospective holdout, not proof of prospective discovery; protect it from repeated tuning. Within development, compare 3-, 5-, and 10-year lookbacks and annual versus quarterly refits. Fail folds with insufficient history rather than silently shortening them.

Starter hyperparameters: `w in {21,63,126}`, `stride in {1,5,21}` subject to `stride<=w`, `k in {2,3,4,5,6}`, metric W1 or W2, 20 restarts. First fix stride 5 to control search cost, compare window lengths and k, then run stride sensitivity. Reject collapsed or unstable solutions and use held-out MMD homogeneity, separation, occupancy, seed stability, and detection delay jointly. Choose the smallest stable k within a prespecified near-best tolerance; no single index establishes true regimes.

A global training-fit affine scaling of returns is optional. Per-window centering/standardization removes location/scale regimes, so reserve it for a named shape-only ablation. Do not winsorize the primary distribution silently; investigate bad ticks and compare a separate robust variant. Full quantile-coordinate standardization alters Wasserstein geometry and is not the default.

## 6. Evaluation and reporting

Run every baseline on the same eligible observations and chronological splits. Include a training-calibrated volatility threshold, mean/std k-means, four-moment k-means, Gaussian mixture, and Gaussian HMM. Use training-only standardization for feature models and filtered probabilities for causal HMM evaluation; full-sequence smoothing or Viterbi decoding is retrospective. The HMM sees individual returns, so point-level and window-level comparisons must define their aggregation and timing explicitly.

For each pair of scalar return samples use the Gaussian-kernel biased statistic

$$\widehat{\mathrm{MMD}}_b^2=\frac{1}{m^2}\sum_{i,j}k(x_i,x_j)+\frac{1}{n^2}\sum_{i,j}k(y_i,y_j)-\frac{2}{mn}\sum_{i,j}k(x_i,y_j).$$

Compute MMD on the same underlying return samples across algorithms, not on one algorithm's features and another's atoms. Fix the kernel bandwidth from training data, with a positive floor and prespecified multi-bandwidth sensitivity. Report median within-cluster and between-cluster values with per-cluster and occupancy-weighted summaries. Empty/singleton clusters receive explicit unavailable diagnostics rather than invented zeros.

Primary real-data diagnostics use disjoint or sufficiently separated held-out window pairs. For uncertainty, block-resample the raw return stream, rebuild windows, and, when estimating total procedure uncertainty, refit the entire pipeline. Report block length and sensitivity; IID bootstrap over overlapping windows is invalid. Do not label these descriptive MMD estimates as significance tests without dependence-aware calibration and correction for model selection.

Synthetic evaluation supplies ground truth: adjusted Rand index, matched balanced accuracy, regime-specific recall, false switches, delay to detect a true switch, and centroid distance to known distributions. Score pure windows separately from transition windows. For simulated online forecasts, map arbitrary cluster IDs to regime names using training labels only. Whole-sequence optimal label matching is acceptable solely as an explicitly retrospective clustering diagnostic.

Reports should show assignment timelines at **window endpoints**, centroid quantile/ECDF overlays, tail zooms, member distributions, cluster sizes, transition and dwell-time tables, MMD distributions, seed/refit stability, novelty versus time, data gaps, and detection-delay distributions. A separate retrospective overlap-vote plot can match the paper. Historical crisis labels provide context and must not become post-hoc ground truth.

Economic validation is optional and downstream: freeze a simple position/risk rule on validation data, shift decisions to an executable bar, and include spread, slippage, turnover, fees, and funding where applicable. Compare to the same rule without regimes. Descriptive separation alone is not evidence of incremental economic value.

## 7. Scale and performance

With `M` windows, `w` atoms, `k` clusters, `I` iterations, and `R` restarts, sorting costs `O(M w log w)` and assignment/update work is approximately `O(R I M k w)`, plus median selection costs for W1. Store `O(Mw + kw)` values. Avoid materializing a full `M x M` distance matrix or a huge `M x k x w` broadcast tensor. Use chunked assignments; for W2, a matrix-product distance calculation is another exact option with numerical cancellation safeguards.

Illustrative scale: 10,000 daily returns at `w=63,stride=5` yield 1,988 windows and approximately 1 MB of float64 atoms. A one-million-window workload at `w=252` needs about 2 GB for atoms alone and should use chunks/memory mapping. CPU-first is appropriate; profile before adding Numba or GPU machinery.

MMD can dominate runtime because each pair costs `O(w²)`. Start with 5,000 recorded evaluation pairs, cache each window's self-kernel mean, and chunk cross-kernel computations. The paper's very large pair counts belong in a separate expensive run. Benchmark on recorded hardware; targets are in the implementation plan and are not measured results.

## 8. Multivariate and temporal extensions

Different questions require different empirical objects:

| Question | Object | Appropriate method |
| --- | --- | --- |
| SPY regime through time | Distribution of SPY returns in one window | Exact 1D W1/W2 |
| Several assets' separate marginal regimes | One distribution per asset per window | Separate models or weighted sum of marginal distances; does not capture dependence |
| Joint market regime | Distribution of synchronized vectors `(r_SPY,r_TLT,r_GLD,...)` | Sliced Wasserstein or multivariate OT |
| Serial-dependence regime | Distribution of lag vectors `(r_t,r_(t-1),...)` | Sliced/path-aware distance; changes sample count and dimension |

For the joint extension, align return intervals and fit per-asset scales on training data. Project each joint sample cloud along fixed seeded unit directions, calculate exact 1D distances, and average p-th powers before taking the p-th root. Persist projections so training and inference share the same metric. Start at 128 directions, benchmark 32/128/512, and report convergence of distances and assignments.

Use **sliced-Wasserstein k-medoids** first. Averages of independently projected quantile functions need not correspond to a realizable joint distribution; labeling them an exact multivariate Wasserstein barycenter is incorrect. K-medoids retains an actual sample-cloud representative. If barycenters are needed, implement a separate particle optimizer for a specified sliced or entropic OT objective, with approximation/convergence diagnostics. Entropic OT and Sinkhorn divergence have different objectives and should not silently replace exact Wasserstein distance. [POT documentation](https://pythonot.github.io/all.html).

Include a dependence-only synthetic test with identical asset marginals and changing correlation; univariate marginal models should fail by construction, while a successful joint extension should improve. Include a serial-order control with identical bags of returns to demonstrate the univariate method's blind spot. These tests establish why an extension is needed rather than relying on more elaborate code as evidence.
