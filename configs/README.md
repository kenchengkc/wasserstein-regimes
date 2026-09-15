# Frozen first study

The first SPY specification was recorded before evaluating 2024–2026 returns.
The input is revised adjusted-close history, not a point-in-time total-return
archive. Availability at exchange close plus one minute is an explicit replay
assumption; adjusted historical prices may have been revised later.

Annual development test folds cover 2019–2023. Each uses expanding training
ending four calendar years before the test year, then three validation years.
For example, the 2019 fold trains through 2015 and validates on 2016–2018.
The final model trains through 2020, validates on 2021–2023, and stays frozen
through the complete 2024–August 2026 holdout. No holdout-based refits or tuning.

K is selected from 2–5 by validation silhouette in full sorted-return W2
geometry, with ties favoring smaller K. All baselines use the same selected K;
this controls state count but does not optimize each baseline individually.
L=63, fitting stride=5, scoring stride=1 and a 99th-percentile novelty threshold
are fixed. Other lengths and strides are sensitivity studies, not selections
based on holdout performance. W1 is robustness; standardized W2 tests shape.

Strict results purge shared raw price intervals at training/validation/test
boundaries, including the price preceding a window's first return. Operational
results score all eligible daily windows. Neither makes overlapping daily test
windows independent. MMD comparisons use separated windows; uncertainty uses
blocks. Seed/bootstrap repeat counts quantify sensitivity, not formal discovery
significance. Future 21-session outcomes are descriptive and never fit inputs.

No result is a trading strategy evaluation. A single ETF and a single historical
holdout cannot establish general incremental economic value.
