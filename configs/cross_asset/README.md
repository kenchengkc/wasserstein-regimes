# Frozen cross-asset replication

This specification was recorded before evaluating QQQ, TLT, GLD or HYG results.
The four ETFs are fitted independently using the original SPY methodology:
63-return windows, fit stride 5, daily scoring, K=2–5 chosen by validation W2
silhouette, 20 empirical restarts, the same nine models, five development test
years (2019–2023), and a fixed 2024–August 2026 holdout. All preprocessing and
model fits remain asset-specific. No pooling or cross-asset parameter selection.

Training starts at each ETF's first available observation. This preserves the
expanding-history procedure but means differences across assets can reflect
history length as well as the asset. The SPY reference retains its original
snapshot and published results; it is not silently downloaded or refitted.
All new datasets use the same adjusted-close provider, requested date range and
cutoff. Actual inception dates, retrieval timestamps and hashes are recorded.
Revised history is not point-in-time data. Asset returns are correlated, so four
ETFs do not constitute four independent statistical replications.

Primary comparison: raw centroid location/scale/shape fractions, agreement with
volatility and rich-feature baselines, occupied states, and shape-only
seed/block-bootstrap stability on validation. Across-time shape stability is
measured by consecutive saved-model refits on common current validation windows
using only observations available before that refit's test period. ARI remains
valid when K changes; centroid displacement is reported only for equal K and
uses minimum-cost matching. This is descriptive refit sensitivity, not a
forecast evaluation. Per-fold occupancy accompanies ARI to expose degenerate
one-state agreement. New summary calculations do not retrain models.

The replication will report continuous evidence, failures and mixed results;
there is no newly selected pass threshold or asset-specific tuning. It does not
establish predictive usefulness merely because shape clusters differ from raw
clusters. Risk forecasts, multivariate OT and trading remain outside this
release. Model or data failures will be reported and diagnosed rather than
silently dropping an asset or changing K to obtain a cleaner result.
