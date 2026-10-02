# Exploratory risk forecasting protocol

Frozen before examining the forecast losses. This is a research comparison, not a trading system. The 2024–August 2026 prices had already been inspected in earlier studies; this assessment is explicitly exploratory. Revised adjusted-close snapshots are not point-in-time data. The [recorded protocol](https://github.com/kenchengkc/wasserstein-regimes/blob/4ce3b1b4fb9f7adbc0bb9bbf6263266a7b2b656c/docs/risk-protocol.md) preserves the original specification.

## Question and frozen choices

Does the existing training-scaled joint sliced-W2 model improve five-session risk forecasts over simple historical and volatility-only controls? Keep its selected K=3, 63-session windows, fitted scales, medoids and projections unchanged. Selection ended December 2023. No refits, tuning or alternative risk specifications after seeing losses.

Use the existing synchronized SPY, QQQ, TLT, GLD and HYG snapshots in that order. Convert their log returns back to simple returns and take an equal-weight daily-rebalanced basket, `r[t] = mean(expm1(log_returns[t]))`. Set `q[t] = r[t]**2`. The target at session t is `y[t] = mean(q[t+1:t+6])`, the next five sessions' average daily squared basket return. Forecast the conditional **second moment**; it equals variance only under a zero conditional mean assumption. This is neither squared five-day cumulative return nor intraday realized variance. No annualization in losses.

Forecast after all components of the current observation are available (session close plus one minute in these snapshots). Target observations must have later availability timestamps. These close-to-close proxies do not imply an executable trade at the just-observed close. Require synchronized, complete daily returns and monotonically increasing availability; reject missing data rather than fill or compress the session clock.

All five methods use exactly the same observation prefix through t, assets, targets and scored origins:

| Method | Forecast of average daily squared return |
| --- | --- |
| `expanding` | Mean of q from the start of common history through t |
| `ewma` | Start with the first 252 q observations' mean; update with lambda=0.94 and q[t] thereafter; constant expected daily risk across the five-session horizon |
| `rolling` | Mean of the latest 63 q observations |
| `regime` | Mean of matured five-session targets whose originating window has the current frozen joint state, shrunk with 252 pseudo-observations at the current expanding mean |
| `volatility_state` | Same matured-target estimator and shrinkage, but use three bins of trailing 63-session mean q; bin boundaries are training-period terciles (through 2020), frozen thereafter |

State estimators use `(state_target_sum + 252 * expanding)/(state_count + 252)`. At t they may update only from origins s with s+5 <= t. Start historical state origins after the first complete 63-return window. Historical assignments using the frozen classifier are retrospective training statistics, never presented as live pre-2024 forecasts. The classifier and volatility bin boundaries do not adapt; state-target averages update causally as outcomes mature. Empty states fall back to expanding risk. Floor all forecasts at 1e-12, record any floor applications, and never floor targets. Require 252 observations before any forecast.

## Assessment and uncertainty

Start requests January 2024, but retain only origins whose entire 63-return raw-price interval begins after the December 2023 calibration cutoff. Score daily origins through August 2026 only when all five future returns are present and available by the fixed as-of timestamp. Save unscored final forecasts with pending targets locally and count them explicitly. Report the common origin range, target maturity range, state occupancy, mapping support, and yearly descriptive slices.

Primary proper loss: `log(forecast) + target/forecast` (QLIKE up to a target-only term). It is finite at zero target and is minimized in expectation by the positive conditional mean of the target. Negative absolute scores are valid and have no percentage interpretation. Report paired differences `regime - baseline`; negative favors regimes. Secondary loss: squared error on the same unannualized second-moment scale. Primary comparator: EWMA. All four comparator rows are reported; do not select a winner or claim confirmatory significance.

Use paired circular stationary-bootstrap draws of the chronological loss differences, 2,000 replicates, seed 2026. Mean block length 63 origins is primary; 20 and 126 are prespecified dependence sensitivities. Preserve all methods' pairing with shared draws. Report pointwise percentile 95% intervals, not p-values or simultaneous/multiplicity-adjusted bands. Bootstrap is conditional on the already selected frozen model and forecasting rule; it does not incorporate model selection, parameter refitting, source revisions or nonstationarity. The short assessment and overlapping targets/windows limit inference even with long blocks.

[Patton (2011)](https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf) motivates losses robust to conditionally unbiased volatility proxies. Here we directly target the conditional mean of squared returns and make the extra zero-mean condition explicit before interpreting it as variance. [Politis and Romano's stationary bootstrap](https://statistics.stanford.edu/technical-reports/stationary-bootstrap) supplies the dependent resampling method; approximate stationarity remains an assumption.

## Reproducibility and stopping rule

The `risk` job uses the content-bound, resumable executor. Its identity fingerprints the frozen bundle, risk config, all five CSVs and acquisition sidecars, numerical sources and dependencies. Forecast-level audit data stay in ignored local artifacts. Published outputs contain only aggregate comparisons, provenance and checksums; no market observations, medoids or per-window states. The worker seals complete outputs and verifies consumed inputs before and after execution.

A negative result is a complete scientific outcome; settings must not be retuned to beat EWMA on this assessment. A confirmatory successor requires a newly locked protocol and genuinely unseen observations, plus point-in-time/source validation. Tail risk, covariance-matrix targets, GARCH, trading allocation and transaction costs are separate extensions, not claims of this study.
