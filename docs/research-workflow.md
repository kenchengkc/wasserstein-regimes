# Reproducing the univariate study

Install Python 3.11 or newer and the research dependencies:

```bash
python -m pip install -r requirements-research.txt
python -m pip install -e '.[research,dev,data]'
export VECLIB_MAXIMUM_THREADS=1
python -m pytest
```

The base package remains NumPy-only; CSV ingestion, model baselines, research
commands and reporting require the `research` extra. Tests include independent
assignment and sklearn oracles, mathematical invariances, causal HMM filtering,
calendar gaps and saved-model prefix invariance.

## Data snapshot

The first study uses 8,454 SPY adjusted daily closes, January 29, 1993 through
August 31, 2026, acquired with yfinance 1.7.0. Its SHA-256 is:

```text
aeb4a4356b6d34fc4c67d7f446e98e8e91d2fc57a48065616137dea379d21b7d
```

The CSV is local at `data/raw/spy_daily_2026-08-31.csv`. Source data is not bundled.
A new snapshot can be acquired with:

```bash
python scripts/download_spy.py --output data/raw/spy_new_snapshot.csv
```

This refuses overwrites. A fresh provider download may differ from the study's
frozen hash. To conduct a new study, copy the configuration and record that new
path/hash rather than relabeling the result as an exact reproduction. Adjusted
history is revised history; it is not a point-in-time data archive. The data
loader validates prices and sessions, preserves missing sessions as gaps and
excludes windows crossing them. It never silently substitutes unadjusted close.

## Frozen temporal specification

See `configs/spy_daily_w2.yaml` and `configs/README.md`. The initial specification
was committed before evaluating the holdout. Development test years are
2019–2023. Each expanding training period is followed by three validation years
and one test year. The final model trains through 2020, validates on 2021–2023,
and remains fixed throughout January 2024–August 2026.

K=2–5 is chosen using validation silhouette in full quantile-space W2 geometry.
All baselines share that K. L=63, fit stride=5, score stride=1 and novelty
percentile 99% are fixed. W1 is a robustness model; shape W2 standardizes each
window using its own population mean and standard deviation.

Strict evaluation purges shared raw-price intervals across boundaries,
including the price preceding the first return. Operational replay allows
already-known observations before the boundary. Strict daily test windows
still overlap with each other. Availability at scheduled exchange close plus
one minute is a declared replay assumption, not verified historical vendor
publication latency.

## Commands and artifacts

```bash
regimes run --config configs/spy_daily_w2.yaml --stage development
regimes synthetic --config configs/spy_daily_w2.yaml
# Open only after development checks and frozen specification:
regimes run --config configs/spy_daily_w2.yaml --stage holdout
regimes report --run RUN_ID
```

`run` includes the synthetic controls and transport benchmark, and writes
configuration, provenance, fitted primary model, all model
prototypes, feature/HMM parameter audit records, per-fold distances and exact
location/scale/shape decompositions, assignments in Parquet and summary metrics.
`checksums.json` verifies saved research inputs on reuse. Source code identity,
configuration, dependency versions and immutable dataset hash determine the
run ID. Reports are generated solely from saved files. The root model and
centroid files describe the latest fold; all earlier primary models are stored
under `models/YEAR/`.

The full-distribution estimator is also usable directly:

```python
from wasserstein_regimes import WassersteinKMeans

model = WassersteinKMeans(metric="w2", n_clusters=3, n_init=20, random_state=42)
model.fit(train_windows)
labels = model.predict(test_windows)
distances = model.transform(test_windows)  # true W2, not squared W2
model.save("model")
restored = WassersteinKMeans.load("model")
```

The W2 objective is sum of squared W2 distances. W1 uses sum of W1 distances
and coordinate medians. Arrays represent all equally weighted observations.
No histogram approximation or multivariate sorting shortcut is used.

## Interpretation limits

- Cluster IDs are local to a fitted model. Pairwise refit matching is saved when
  K agrees; a changed K does not admit a bijective alignment.
- ARI=1 can occur when both methods assign every test window to a single state.
  Always inspect occupancy alongside agreement.
- Centroid separation describes learned prototypes. Within-assignment shape
  variation is not evidence that shape drove the partition.
- Feature/GMM/HMM novelty is distributional distance to their prototypes; their
  own assignment rules need not choose the nearest W2 prototype. HMM prototypes
  are fitted Gaussian emission quantiles, and assignments use forward filtering.
- Novelty percentiles are validation-calibrated ranks, not probabilities of a
  crisis. Margins are relative distance gaps, not class probabilities.
- Stability checks cover seeds, return-block resampling and refits. Bootstrap
  windows are rebuilt after resampling returns, including block joins. Seed
  stability alone does not establish a real regime.
- Future 21-session return, volatility, drawdown and downside deviation are
  evaluation-only. Their block percentile intervals are descriptive, with
  limited effective samples; no trading or predictive improvement is claimed.
- The stationary overlap null preserves the empirical marginal through IID
  resampling while removing serial dependence. Its persistence distribution
  can be wide or degenerate. Twenty draws are sensitivity evidence, not a
  calibrated hypothesis test.
- Heavy-tail and skew controls match population moments. Only the special
  disjoint 300-atom fixture matches the first four empirical moments exactly.
- Benchmark kernels receive sorted vectors; sorting/preparation is not included
  in their timing. Python-traced memory is not full process peak RSS.

Cross-asset replication, multivariate OT, path-aware distributions and
regime-conditioned forecasting remain separate future phases. The original
[design](design.md) and [implementation plan](implementation-plan.md) describe
that longer roadmap; this release deliberately implements the univariate
research slice.
