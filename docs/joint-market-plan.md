# Joint market implementation plan

**Goal:** Execute the frozen exploratory panel study and publish auditable aggregate evidence.
**Spec:** [Protocol](joint-market-protocol.md).
**Architecture:** Strict configuration and hashed panel preparation feed saved baseline/joint models. Development and assessment artifacts are separate sealed directories; assessment loads the development models without fitting.
**Stack:** Existing Python research extras. GPL-3.0-only. No new dependencies or market downloads.
**Execution:** Inline implementation and one independent review; tests precede new behavior.

## Review focus

Calendar equality does not imply matching return horizons. Data after training must not affect scaling or fitting. HMM predictions must use filtering. Model files must retain all preprocessing state. Assessment must not silently retrain or accept a missing/mismatched development run.

- [x] Implement panel_baselines.py: marginal/covariance/correlation features with train-only feature transforms; full-covariance Gaussian HMM with multistart diagnostics, forward recursion and NPZ state. Test hand-derived feature geometry, causal prefixes, and saved prediction equivalence.
- [x] Implement joint_market.py: validate configuration including unknown keys, verify all snapshot hashes/sidecars, explicitly crop common history, build strict windows and training-only scales, freeze K on validation, compute sensitivity and aggregate diagnostics. Test temporal invariance, interval separation and stage guard. Register joint-study CLI.
- [x] Review before research execution. Freeze code; run development, inspect failure/convergence diagnostics, then explicit assessment with saved-model inference. Verify sealed artifacts and all counts. Publish aggregate JSON/report and truthful conclusions. Run full tests, documentation checks and CI; update the existing PR to describe its complete scope.

No silent retuning or new untouched-holdout claim. Source sensitivity and full resampling/refit/null validation remain explicit follow-up work.
