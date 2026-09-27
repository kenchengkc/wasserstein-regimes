# Phase 6 implementation plan

Spec: [frozen risk protocol](risk-protocol.md). Base: phase-5 branch `feat/resumable-experiments`; the risk PR is stacked until its dependency merges.

1. Implement NumPy forecasting and paired uncertainty kernels. Tests precede code: hand-calculated maturity/shrinkage, cold/empty states, EWMA recursion, future-prefix invariance, zero-target losses, invalid inputs and deterministic paired bootstrap.
2. Implement contract-checked panel evaluation and `risk` job integration. Test fixed model inference, strict post-calibration origins, later target availability, pending tails, common method rows, missing/alignment rejection, config/source/input identity and a real resumed worker using synthetic snapshots.
3. Obtain one independent scientific and implementation review before running empirical losses. Fix important issues with regression tests. Run the whole suite and strict documentation build.
4. Execute the frozen study, verify cache reuse and sealed inventory, publish aggregate evidence, update results/roadmap/workflow, and open a stacked PR. Preserve the protocol and report all unfavorable comparisons.

Review focus: future labels versus matured targets; exactly shared information sets; return convention and target interpretation; late availability and gaps; state calibration chronology; bootstrap pairing/dependence; output identity and market-data publication boundaries.
