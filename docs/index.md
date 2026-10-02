<p align="center">
  <img src="assets/brand/wasserstein-regimes-logo.png" alt="Wasserstein Regimes logo" width="144" height="144">
</p>

# Wasserstein Regimes

**What market-state information is lost when a return window is compressed into volatility or a few moments?**

This research compares full empirical return distributions with volatility, moment and dependence baselines. Synthetic controls establish what the representations can distinguish; chronological market studies test their stability and forecasting value.

## Marginal clusters mostly reflect scale

In SPY, **97.87%** of raw-W2 training-centroid separation is scale. Holdout assignments closely agree with volatility-only clustering (**ARI 0.990**). Independent QQQ, TLT, GLD and HYG studies find **95.27–97.53%** scale contribution. Shape stability is mixed; TLT and HYG occupy only one raw holdout state.

Full distributions separate exact matched-four-moment synthetic laws. That capability does not establish incremental market information or predictive value from shape. ARI measures agreement, not accuracy; identical single-state assignments can produce ARI 1.

## Joint information is detectable; empirical partitions remain fragile

Joint sliced-W2 medoids distinguish dependence changes missed by marginal representations, including a constructed control with identical covariance. Correlation changes are also detected by covariance clustering, so correlation recovery alone establishes no distributional advantage.

In the market study, return-level resampling gives mean validation **ARI 0.518**. A seeded **1% extreme training contamination** scenario collapses clean validation assignments to one state. Stationary nulls produce persistent clusters and novelty flags. These are sensitivity measurements, not confidence intervals or calibrated market hypothesis tests.

## Risk forecasting shows no demonstrated advantage

Across **600** scored five-session basket-risk forecasts, joint regimes have higher average primary QLIKE loss than expanding history, EWMA, rolling history and volatility-only states. The regime-minus-EWMA difference is **+0.1715**, with a paired 95% interval of **[−0.0217, +0.4816]**. Lower loss is better; the interval establishes neither superiority nor equivalence.

The target is the next five sessions' average daily squared simple return of an equal-weight SPY/QQQ/TLT/GLD/HYG basket: a conditional second moment, interpretable as variance only under zero conditional mean. Paired bootstrap intervals condition on the selected model and frozen forecasting rule; they exclude selection uncertainty and do not guarantee coverage under nonstationarity.

![Exploratory five-session risk forecast comparison, with paired uncertainty intervals](assets/risk-comparison.png)

## Published studies

| Study | Evidence |
| --- | --- |
| [Univariate market regimes](research-results.md) | SPY geometry, baseline agreement, overlapping-window nulls and synthetic recovery |
| [Cross-asset replication](cross-asset-results.md) | Independent QQQ/TLT/GLD/HYG fits, occupancy and shape stability |
| [Joint dependence controls](joint-results.md) | Representational capability beyond marginals and covariance; measured computation |
| [Joint market panel](joint-market-results.md) | Six models on 605 shared windows; conditional window-refit mean ARI 0.521 |
| [Robustness and failure modes](joint-validation-results.md) | Return resampling, stationary nulls, rare transitions and contamination |
| [Frozen inference and execution](execution-results.md) | 690/690 validation labels reproduced; all 11 jobs resume from verified artifacts |
| [Risk forecasting](risk-results.md) | Frozen target, all baseline comparisons and dependence sensitivities |

Each study reports limitations, source identities and aggregate artifact checksums. Execution measurements establish reproducibility and resource use, with no implication for regime stability or prediction.

## Methods and research context

Exact one-dimensional W2 clustering uses every order statistic and mean quantile barycenters; W1 uses median barycenters. Joint models use finite-projection sliced W2 and sampled observed-window medoids, an approximation to sliced geometry rather than a full multivariate W2 solver. Both representations discard temporal order within each window; chronological assignment does not recover it.

Read the [methods](methods.md), frozen [market](joint-market-protocol.md), [robustness](joint-validation-protocol.md) and [risk](risk-protocol.md) protocols, [Research agenda](research-roadmap.md) and [motivating paper review](paper-review.md). Next questions include temporal-order controls, source validation and confirmation on newly reserved observations.

## Reproduce the evidence

- [Univariate study workflow](research-workflow.md)
- [Frozen scoring and experiment workflow](execution-workflow.md)
- [Risk forecasting workflow](risk-workflow.md)
- [Data sourcing, schemas and availability assumptions](data.md)

Market CSVs, observed medoids and per-origin audit rows remain local. Fresh downloads may differ from frozen hashes and cannot establish exact reproduction. Numerical sources, dependencies, configuration and input identities bind the saved evidence. The estimators require only NumPy; research commands need the research dependencies.

## Scope and data rights

The joint market and risk studies reuse a previously inspected 2024–August 2026 period and are explicitly **exploratory**. Revised adjusted-close histories are not verified point-in-time archives. Overlapping windows, correlated assets and sparse state occupancy limit inference; cross-provider validation remains open. No trading profitability or executable-close claim follows from these results.

Motivated by Horvath, Issa and Muguruza's [Clustering Market Regimes using the Wasserstein Distance](https://arxiv.org/abs/2110.11848v1). These independent daily studies do not reproduce the paper's hourly experiment.

Original code and documentation are **GNU GPL v3 only** (`GPL-3.0-only`); see [LICENSE](https://github.com/kenchengkc/wasserstein-regimes/blob/main/LICENSE). Third-party data and papers retain their own terms.
