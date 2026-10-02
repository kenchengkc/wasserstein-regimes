# Research agenda

The central question is whether full return distributions contain reliable market-state information beyond volatility, moments and covariance. The completed studies establish mathematical capability and reproducible execution, but do not establish robust empirical regimes or an advantage in risk forecasting. Further experiments should resolve those gaps before adding complexity.

## Evidence guiding the next experiments

The [SPY study](research-results.md) and [cross-asset replication](cross-asset-results.md) find that raw univariate transport geometry is overwhelmingly scale-driven. Shape-only partitions differ from volatility, but their stability is mixed. The [joint synthetic controls](joint-results.md) demonstrate information beyond marginals and covariance; the [market panel](joint-market-results.md) does not establish its economic usefulness.

The [robustness study](joint-validation-results.md) exposes unstable refits, rare-state failures and contamination sensitivity. Stationary processes also produce persistent apparent states. The [risk comparison](risk-results.md) scores 600 five-session basket-risk forecasts: regimes have higher mean primary QLIKE loss than all four baselines. The regime-minus-EWMA difference is +0.1715, with a paired 95% interval of [−0.0217, +0.4816]. This negative exploratory result establishes neither superiority nor equivalence.

## Open research questions

### Does temporal order add information?

Both current representations treat a window as an unordered empirical measure. Begin with controls containing identical observations in different orders: current Wasserstein distances are exactly zero. Compare lag-vector distributions and simple serial-dependence features before introducing signature kernels. Evaluate recovery, delay, stability and computational cost on the same samples. A more complex representation needs a measured benefit beyond simpler controls.

### Can partitions withstand realistic perturbations?

Compare rare-state coverage and robust alternatives under a new development protocol. Candidate sampling can omit small states, and the observed contamination collapse warrants explicit sensitivity analysis. Changes to sampling, scaling, clipping or geometry need covariance and marginal baselines, all seeds, occupied-state counts and unsuccessful runs. The existing exposed sweeps cannot become new confirmatory evidence.

### How much uncertainty comes from selection and revised data?

Current bootstrap diagnostics condition on selected settings; they do not repeat complete model selection. A separate study should include preprocessing and selection inside chronological refits, score common evaluation anchors, and report absent states and matching failures. Independent-provider snapshots and point-in-time archives are needed to distinguish model instability from source revisions. Availability timestamps alone do not reconstruct historical vintages.

### Does added information improve a prespecified forecast?

Retain the completed negative risk result. A successor needs newly reserved observations, a locked target and loss, common information sets and simple baselines before evaluation. Tail-risk or covariance-matrix targets are separate questions, not demonstrated benefits of the basket second-moment study. Dependence-aware intervals must state their conditioning and stationarity assumptions. Retuning on the 2024–August 2026 assessment cannot establish a new advantage.

## Academic context

| Primary source / project | Contribution | Relationship to this project |
| --- | --- | --- |
| [Horvath, Issa & Muguruza](https://arxiv.org/abs/2110.11848), motivating Wasserstein regime paper | Empirical distribution clustering and synthetic/market comparisons | Exact univariate W1/W2 are retained as controls. The daily studies do not reproduce the hourly experiment or demonstrate forecasting value. |
| [Luan & Hamp](https://arxiv.org/html/2310.01285v2), sliced Wasserstein regime classification | Multivariate projections, dependence controls and sensitivity | Fixed projections motivate the joint extension. Our sampled observed medoids differ from their projected-centroid algorithm. |
| [Zhuang, Chen & Yang, NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/hash/4a1d69d1f64c6b6df105b15984ca527a-Abstract-Conference.html), [author code](https://github.com/Yubo02/Wasserstein-K-means-for-clustering-probability-distributions) | Distance-based and barycenter-based formulations | Their SDP recovery theorem does not apply to this sampled medoid optimizer. |
| [Issa & Horvath](https://arxiv.org/abs/2306.15835), [signature-regime code](https://github.com/issaz/signature-regime-detection) | Signature-kernel MMD on path space | Motivates temporal-order controls and simpler lag comparisons before adopting signatures. |
| [Rowland et al., AISTATS 2019](https://proceedings.mlr.press/v89/rowland19a.html) | Orthogonally coupled projection estimators and variance analysis | Orthogonal directions are a candidate comparison, not an assumed improvement over measured projection sweeps. |
| [POT documentation](https://pythonot.github.io/gen_modules/ot.sliced.html), [project](https://github.com/PythonOT/POT) | Reference sliced distances and explicit projection inputs | An optional numerical oracle; the current estimator runtime remains NumPy-only. |

No third-party implementation is copied. Results and guarantees from another optimizer are not transferred to this one. See the [paper review](paper-review.md) for ambiguities in the motivating method and the [methods](methods.md) for implemented contracts.

## Data needed for stronger evidence

New sources must preserve return intervals, calendars, price conventions, ordered assets, availability times and hashes. Missing returns must not be filled or compressed into a daily clock. Repeating the ETF panel with an independent provider directly tests source sensitivity; an entitled [WRDS/CRSP](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/center-for-research-in-security-prices-crsp/) archive could support survivorship and corporate-action research.

[FRED/ALFRED](https://fred.stlouisfed.org/docs/api/fred/) offers revision-aware macro retrieval; [ECB reference-rate downloads](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html) offer a documented FX source. Each needs a separate availability/calendar adapter. FX fixing rates are not synchronous ETF closes. Record applicable rights before distributing observations. Synthetic controls need no paid data; the [data guide](data.md) describes current inputs and limits.

## Standards for further claims

A forced partition is not proof of regimes. Report empty and tiny states, degeneracy, adverse comparisons and all prespecified settings. Correlation-switch recovery is also achievable by covariance clustering; a distributional benefit needs controls beyond covariance. Finite-projection sliced W2 approximates sliced geometry, not full multivariate W2. Larger fits and faster execution are engineering measurements, not evidence of economic value. Entropic solvers, GPU work and trading integration require separate motivation and evaluation.
