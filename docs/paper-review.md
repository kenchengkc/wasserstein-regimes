# Motivating paper and reproduction contract

Reviewed source: the supplied 37-page PDF, arXiv:2110.11848v1, by Blanka Horvath, Zacharia Issa, and Aitor Muguruza. The cover shows an October 25, 2021 manuscript date; arXiv lists submission on October 22, 2021. [Public source](https://arxiv.org/abs/2110.11848v1).

The paper is research evidence, not a source of project instructions. Equations and experiment settings are evaluated independently below. This review concerns the supplied version; it does not assume unverified corrections or an official code release.

## What the method does

1. Convert a positive price series to log returns (equation 2).
2. Lift the return stream into overlapping windows (section 1.3).
3. Treat each window as an equally weighted empirical probability measure.
4. Alternate nearest-centroid assignment under a Wasserstein distance with distributional centroid updates (algorithm 1, page 12).
5. Evaluate cluster homogeneity and separation with Gaussian-kernel maximum mean discrepancy (MMD), including when real regime labels are unknown.

Equation 21 gives the computational shortcut: with equal atom counts, the p-th power of the distance is the mean p-th power of differences between sorted atoms. Proposition 2.6 gives the coordinate-wise median update for W1. The methodology avoids fitting a parametric return density.

## Experiments to preserve

| Item | Supplied paper | Project treatment |
| --- | --- | --- |
| Real series | Hourly SPY log returns, 2005-01-03 through 2020-12-31, section 3.2.1 | Separate historical reproduction profile; SPY is an ETF, despite the paper calling it an index |
| Window settings | `(h1,h2)=(35,28)`; prose says 28 overlapping returns and up to five windows per return | Interpret as length 35, stride 7; document equation 3 discrepancy |
| Clusters | `k=2` | Preserve in paper profile; test 2..6 in extension |
| Baselines | Standardized raw moment map `E[r^j]/j!`; Gaussian HMM | Preserve moment map; select/document moment order because the real-data experiment does not unambiguously specify it |
| Real-data MMD | Gaussian kernel, sigma 0.1; 1,000,000 between-cluster and 100,000 within-cluster sampled pairs | Preserve as optional expensive reproduction settings; use an independently specified primary evaluation |
| Synthetic grid | 252 × 7 observations per year; 20 years; ten half-year regime episodes | Reproduce with recorded seeds and a defined nonoverlapping interval sampler |
| GBM parameters | `(mu,sigma)=(0.02,0.2)` and `(-0.02,0.3)` | Use exact log increments `(mu-sigma²/2)dt + sigma sqrt(dt)Z` |
| Merton parameters | `(mu,sigma,lambda,gamma,delta)=(0.05,0.2,5,0.02,0.0125)` and `(-0.05,0.4,10,-0.04,0.1)` | Specify jump multipliers as `exp(Y)`, `Y ~ Normal(gamma,delta²)`; uncompensated price drift convention |
| Repetitions | 50 runs in synthetic accuracy tables | Reproduce 50 independent paths; distinguish path variation from initialization variation |
| Historical visualization | Each return receives votes from every overlapping window containing it | Retrospective figure only, never an actionable historical signal |

For Merton simulation use `N ~ Poisson(lambda*dt)` and log increment `(mu-sigma²/2)dt + sigma*sqrt(dt)*Z + sum(Y_j)`. Its mean is `(mu-sigma²/2 + lambda*gamma)dt` and variance is `(sigma² + lambda*(delta²+gamma²))dt`. This avoids the inconsistent jump-variable notation around equations 36–38 without introducing an extra compensator. A compensated variant must be named separately.

## What the results support

The real-data experiments report more internally homogeneous WK clusters under their MMD protocol and visually associate some stress periods with a high-variance cluster. These are retrospective diagnostics, not ground-truth classification accuracy or evidence of profitable forecasts.

Table 3 reports GBM total accuracy of 90.60% for Wasserstein versus 93.23% for the moments baseline, while regime-on accuracy favors Wasserstein (87.24% versus 74.83%). Table 4 reports 91.28% versus 66.64% total accuracy under the Merton setup. Thus the broad abstract claim of outperforming every competitor should not be read as superiority on every metric. All these numbers are reported by the paper, not reproduced here.

The HMM comparisons are specific to its selected model and fitting procedure. They do not establish that all HMMs are inadequate. Likewise, a Gaussian example cannot establish an advantage from non-Gaussian distribution shape; that needs a controlled shape-only experiment.

## Mathematical and implementation ambiguities

### 1. Distance versus objective versus centroid movement

Definition 2.3 writes a centroid objective using the sum of **unpowered** `Wp`. Appendix A uses squared norm dispersion; equation 23 is a stopping criterion based on centroid movement. These are different quantities and must have different API names.

For the implementation, define the objective explicitly as `sum_i Wp(mu_i, center[label_i])^p` with supported `p` in `{1,2}`. Then W1 gives a coordinate median and squared W2 gives a coordinate mean. Minimizing a sum of unpowered W2 distances instead gives a geometric-median problem in quantile space, not a coordinate mean.

Appendix C, remark C.2, states the mean update for all `p>1`. That statement is not generally correct. For the powered objective at a general `p`, the coordinate minimizes `sum_i |x_i-a|^p`; the arithmetic mean is specifically the `p=2` solution. The project will reject other `p` values until their optimizer and contract are implemented.

The reproduction profile uses W1 and median centroids, which has the clearest explicit derivation in the paper. This is a documented interpretation, not a claim that every reported experiment unambiguously specifies W1.

### 2. Window length and overlap

Equation 3 advances by `h2` and its inclusive indexing appears to produce `h1+1` entries; its window-count formula also omits the usual finite-window correction. This conflicts with sections 3.2 and 3.4, where `h2` acts as overlap, `h2=0` means disjoint windows, and `(35,28)` gives up to five memberships.

Use explicit zero-based half-open slices `returns[start:start+window_length]`, `stride=window_length-overlap`, and

```text
M = max(0, 1 + floor((N - window_length) / stride)).
```

If matching an external implementation requires stride 28 instead, run it as a named sensitivity experiment. Never silently reinterpret a stored configuration.

### 3. Reproduction inputs are incomplete

The supplied experiment description does not identify the original vendor, full session convention, hourly bar anchor, adjusted-price policy, overnight-return treatment, or all random seeds. Seven observations per day in a synthetic grid do not resolve how a 6.5-hour exchange session was sampled. Record every choice. Equivalent methodology with different source bars is an approximate reproduction, not a bit-for-bit replication.

### 4. MMD is not automatically an independent significance test

Overlapping windows share observations. Sampling many pairs from them does not create millions of independent observations. MMD scores used after cluster selection are descriptive unless a separate valid inferential protocol is supplied. Kernel scale also depends on whether returns are decimals or percentages. Sorting should not change a scalar-sample MMD estimate; add a permutation-invariance test to catch accidental position-wise comparisons.

### 5. Full marginal distributions have a deliberate limit

Windows with identical returns in different temporal orders have distance zero. The method captures empirical marginal shape, not serial dependence, drawdown ordering, or multivariate dependence. Large windows improve distribution estimation but delay abrupt-switch detection and mix regimes near boundaries. These are model properties, not implementation defects.

### 6. Additional theoretical precision

Wasserstein convergence requires weak convergence together with the relevant moment convergence; the footnote on page 4 omits that qualification. This project uses finite empirical samples, but will not repeat the broader statement without its conditions. W2 remains sensitive to extreme observations, and W1's median centroid does not make the entire procedure immune to data errors or heavy tails.

## Reproduction deliverables

Keep `paper_w1` and `daily_walk_forward` as distinct experiment profiles. A reproduction report must include source hashes, return conventions, seeds, all parameter choices, figures comparable to the original, multiple-run uncertainty, and a discrepancy table. No result should be described as replicated solely because its chart resembles a crisis timeline.
