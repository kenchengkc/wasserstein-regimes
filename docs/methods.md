# Methods and research contracts

The package clusters completed return windows using their empirical distributions. Univariate W1/W2 geometry is exact for equal-size, equally weighted samples; the joint engine uses fixed projections and sampled observed prototypes. These methods describe recurring distributional conditions. Their scientific value is assessed through the [univariate](research-results.md), [cross-asset](cross-asset-results.md), [joint synthetic](joint-results.md), [joint market](joint-market-results.md), [robustness](joint-validation-results.md) and [risk](risk-results.md) studies.

## Empirical return distributions

For positive adjusted daily prices, returns are close-to-close log differences,

$$r_t=\log P_t-\log P_{t-1}.$$

A trailing window of $w$ returns ending at $t$ defines the equal-mass empirical measure

$$\mu_t=\frac{1}{w}\sum_{j=0}^{w-1}\delta_{r_{t-j}}.$$

Every observation is retained, including duplicates and extremes. Sorting the returns gives $q_t$, the empirical quantile representation on $w$ equal-mass intervals. No density estimate or histogram is required. Sorting discards within-window temporal order: permutations of the same returns have distance zero.

Window length and stride are explicit. Longer windows improve distribution estimation but mix states near transitions and delay detection. The endpoint labels the completed window; it does not identify when an underlying change began.

## Exact univariate W1 and W2

For two windows with the same atom count,

$$W_p(\mu_i,\mu_j)^p=\frac{1}{w}\sum_{a=1}^{w}|q_{i,a}-q_{j,a}|^p,\qquad p\in\{1,2\}.$$

`WassersteinKMeans` minimizes the powered objective

$$J_p=\sum_i W_p(\mu_i,\nu_{z_i})^p.$$

For W1, each centroid coordinate is the median of the corresponding member-window quantiles. For squared W2, it is their mean. Both preserve nondecreasing quantiles. These centroids are barycenters for the stated objective, rather than pooled mixtures of member returns. The vector $q/\sqrt{w}$ is an exact Euclidean embedding for W2; standardizing individual quantile coordinates would change that geometry.

The estimator alternates nearest-centroid assignments and centroid updates across seeded restarts, retaining the smallest final objective. It records objective history, iteration count and convergence status. This is a local optimization procedure with no global-optimum guarantee. `transform` returns true distances; `inertia_` is their sum for W1 and the sum of squared distances for W2. Inputs and fitted centers must have matching atom counts.

The shape-only W2 ablation centers and scales each window by its own population mean and standard deviation; constant windows become zero vectors. It intentionally removes location and scale information. Raw W2 diagnostics also decompose squared distance into location, scale and standardized-quantile shape terms. Their [measured contributions](research-results.md) are descriptions of the fitted geometry, not forecasts.

## Joint sliced-W2 medoids

Joint windows retain synchronized return vectors in an array of shape `(windows, atoms, assets)`. Asset order is fixed. The builder requires matching sessions, return basis and input-price intervals, and excludes a whole window if any component is missing or its price chain is discontinuous. Matching dates alone is insufficient; no implicit joining or filling occurs.

For positive asset scales $s$ and fixed unit directions $\theta_1,\ldots,\theta_R$, project each vector as $\theta_r^\top(x/s)$ and sort each projection's atoms separately. With projected quantiles $q^{(r)}$, the squared distance is

$$d_R(\mu,\nu)^2=\frac{1}{Rw}\sum_{r=1}^{R}\sum_{a=1}^{w}\left(q_{\mu,a}^{(r)}-q_{\nu,a}^{(r)}\right)^2.$$

This calculation is exact for the saved finite directions. Random directions approximate sliced W2; the result is not full multivariate W2, and finite directions can fail to distinguish different joint distributions. In one dimension it reduces to exact empirical W2. Projections and scales are copied and frozen during fitting. Empirical study scales are fitted on unique training returns, never on duplicated rolling windows or prediction inputs; raw joint models use scales of one.

`SlicedWassersteinKMedoids` samples candidate windows, builds their squared-distance matrix, and alternates assignments with within-cluster medoid selection on those candidates. Restarts are ranked by the full-training squared-distance objective, scored in chunks. The returned prototypes are actual observed joint windows with saved source indices. Candidate sampling can miss rare states. This is an approximate sampled medoid optimizer, not a multivariate barycenter solver or a globally optimal clustering algorithm. See the [joint study protocol](joint-market-protocol.md) for frozen budgets and baselines.

Joint distributions capture contemporaneous dependence while still discarding row order within a window. Dependence-only controls establish the marginal methods' information limit; correlation-switch recovery alone does not establish an advantage over covariance clustering.

## Data and temporal validity

The daily loader validates positive prices, duplicate dates and exchange sessions. Missing prices create unavailable return intervals rather than multi-day returns disguised as daily observations. Complete windows carry endpoint, raw-price interval and availability metadata. A $w$-return window uses $w+1$ prices, including the price preceding its first return. Daily snapshot availability is assumed to be session close plus one minute; joint-window availability is the latest availability across every component observation. See [data conventions](data.md).

Chronological studies fit centroids and feature transforms on training data, select model choices on validation data, and preserve the resulting state for assessment. Strict splits purge shared raw-price inputs across partitions. The univariate operational replay permits known pre-cutoff returns inside newly completed test windows and is reported separately because those windows remain statistically dependent on earlier data. Overlapping rows do not represent independent observations.

HMM comparisons use forward filtering for scored assignments. Whole-sequence smoothing and retrospective overlap-vote displays do not provide causal signals. Revised adjusted-close snapshots are not point-in-time histories; an availability filter cannot remove retrospective revisions. The joint market and risk assessment periods had already been inspected and remain explicitly exploratory in their [market](joint-market-protocol.md) and [risk](risk-protocol.md) protocols.

Frozen joint scoring validates asset order, return conventions, source identities and saved dimensions. It keeps complete windows whose raw-price start is after training and whose inputs are available by the explicit `as_of` cutoff. It does not refit or estimate new scales. Scores within the novelty-calibration period are marked as retrospective calibration reuse. Details are in the [execution workflow](execution-workflow.md).

## Evaluation and interpretation

Univariate distribution diagnostics use biased Gaussian-kernel squared MMD on scalar return observations, with bandwidth set from training-return scale. Reported pairs are chronologically separated windows. The shape-only ablation instead uses standardized window observations and bandwidth one, so its MMD values have a different geometry. These diagnostics are descriptive and do not supply independent significance tests after clustering and model selection.

Stability studies compare assignments on common validation anchors across seeds, refits and resampled training histories. Return-block resampling rebuilds windows; synchronized vector resampling shares indices across assets. Block joins create synthetic boundaries. The [robustness protocol](joint-validation-protocol.md) distinguishes return resampling from conditional window-block resampling and records fixed model choices. These comparisons condition on selected settings and do not measure full selection uncertainty. ARI measures partition agreement, not accuracy without known truth; empty and single-state solutions remain visible.

The exact matched-four-moment control uses support $\{0,1,2,3,4,5\}$ in fixture units and counts

$$c_\pm=50\pm3[1,-5,10,-10,5,-1]$$

over 300 atoms. The signed fifth finite difference annihilates powers zero through four, so the two positive, normalized empirical laws share their first four moments while retaining different full distributions. The stochastic control samples the same laws with support scaled by $0.01$; finite samples need not match moments exactly. [Synthetic results](research-results.md) distinguish this information-loss counterexample from statistical power or market evidence. Recovery controls separate pure windows from transitions and learn state-label mappings from training truth only.

Cluster IDs are arbitrary, and recurring labels need not form contiguous episodes. A high-dispersion cluster is not automatically a bear market. Distance margins and validation-calibrated novelty are geometric diagnostics, not state probabilities or crisis probabilities. Persistent clusters under stationary nulls, collapsed states and unfavorable comparisons are part of the evidence. The [risk study](risk-results.md) found no demonstrated forecasting advantage over simple baselines; descriptive separation does not establish economic value.

## Reproducibility and memory scope

Saved models retain numerical state and configuration in versioned JSON/NPZ or NPZ archives loaded without pickle, with schema, dimension and finite-value checks. Research artifacts record configuration, snapshots, code and dependency provenance, metrics and checksums. Reports consume saved evidence. Joint assessment requires verified development artifacts and loads saved models without fitting; frozen bundles bind their original numerical sources and dependencies. Observed market medoids contain source observations and remain local with per-window audit data; public evidence is aggregate.

Univariate distances allocate an `(N, K)` result and chunk atom differences, avoiding an `(N, K, w)` temporary. Joint fitting retains the `(N, w, D)` input tensor; conversion to float64 may add a full copy. With $M$ candidates, candidate features require $O(MwR)$ storage and their matrix $O(M^2)$, rather than a full $N$-by-$N$ training matrix for fixed $M$. Scoring projects batches, while `transform` still returns $O(NK)$ distances. This is bounded workspace around resident inputs, not out-of-core training.

The resumable executor bounds worker and numerical-library thread counts and seals completed outputs before publishing success. Recovery occurs at job boundaries, not within an unfinished model fit, and locking assumes local Linux/macOS filesystems. Python-traced allocations and process peak RSS measure different things; per-worker RSS peaks cannot establish simultaneous machine-wide usage or scientific benefit. Consult the [reproduction workflow](research-workflow.md), [execution workflow](execution-workflow.md) and [measured execution results](execution-results.md) for commands, integrity behavior and benchmark scope.
