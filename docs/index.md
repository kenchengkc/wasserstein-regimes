# Wasserstein Regimes

**Question:** What market-state information is lost when a return window is compressed into volatility or a few moments?

**Measured result:** In the frozen SPY study, **97.87% of raw-W2 centroid separation is scale**. Strict holdout assignments agree closely with volatility-only clustering (**ARI 0.990**). Full distributions separate exact matched-four-moment synthetic laws, but this study does not establish added predictive value from shape.

![What drives W2 separation](assets/components.png)

Rolling returns → empirical distributions → Wasserstein geometry → distributional k-means → chronological regime assignments.

Exact one-dimensional W2 clustering of equal-size empirical distributions is Euclidean k-means over **all order statistics**, rather than selected moments. W2 uses mean quantile barycenters; W1 uses median barycenters. Sorting intentionally discards within-window temporal order.

**Cross-asset replication:** QQQ, TLT, GLD and HYG show 95.27–97.53% scale contribution to raw centroid separation. Shape stability is mixed; TLT and HYG occupy only one raw holdout state. Read the [cross-asset results](cross-asset-results.md).

**Joint research extension:** Fixed-projection sliced W2 and sampled medoids detect dependence that marginals miss, with a fixed candidate memory budget. Read the [academic comparison and roadmap](research-roadmap.md) and [measured synthetic results](joint-results.md).

## Completed research release

- Frozen daily SPY snapshot, five development folds and a fixed 2024–August 2026 holdout.
- Raw W2, W1 and shape-only W2; volatility, mean/volatility, moments, rich features, GMM and causal HMM baselines.
- Exact location/scale/shape decomposition, seed/block/refit stability, overlap-null persistence, validation-calibrated novelty and evaluation-only future outcomes.
- Five synthetic controls, window-length power/delay studies, measured kernels, immutable artifacts and saved-artifact HTML reports.
- 142 tests, including independent numerical oracles and a full price-prefix leakage regression.

Read the [measured results](research-results.md), [reproduction workflow](research-workflow.md), [data sourcing](data.md), [paper review](paper-review.md), [design](design.md) and [future implementation roadmap](implementation-plan.md). Aggregate evidence is saved in [results](https://github.com/kenchengkc/wasserstein-regimes/tree/feat/distributional-evidence/results).

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-research.txt
python -m pip install -e '.[research,dev,data]'
export VECLIB_MAXIMUM_THREADS=1
python -m pytest
regimes run --config configs/spy_daily_w2.yaml --stage development
regimes run --config configs/spy_daily_w2.yaml --stage holdout
regimes report --run RUN_ID
```

The frozen CSV is deliberately excluded. A new download may differ because adjusted history is revised; follow the workflow before claiming an exact reproduction. `run` includes synthetic controls and the numerical benchmark. `report` never refits models.

```python
from wasserstein_regimes import WassersteinKMeans

model = WassersteinKMeans(metric="w2", n_clusters=3, n_init=20, random_state=42)
model.fit(train_windows)
labels = model.predict(test_windows)
distances = model.transform(test_windows)  # true W2 distances
```

The base estimator remains NumPy-only. Research commands require the `research` extra. Multivariate OT, live inference and trading remain outside this release.

## Paper and license

Motivated by Horvath, Issa and Muguruza, [Clustering Market Regimes using the Wasserstein Distance](https://arxiv.org/abs/2110.11848v1). This is an independent empirical study, not a claim to reproduce the paper's hourly experiment.

Original code and documentation are **GNU GPL v3 only** (`GPL-3.0-only`); see [LICENSE](https://github.com/kenchengkc/wasserstein-regimes/blob/main/LICENSE). Third-party data and papers retain their own terms.

## Static documentation deployment

Vercel serves MkDocs output with `framework: null`, `requirements-docs.txt`, and output directory `site/`. The research package needs no HTTP entrypoint. Build with `python -m mkdocs build --strict`; the documentation build neither downloads data nor runs research.
