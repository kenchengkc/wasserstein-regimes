# Joint distribution implementation plan

**Goal:** Deliver the first tested joint-distribution research engine and measured synthetic evidence.
**Architecture:** Separate temporal window alignment from finite-projection geometry and sampled medoid optimization. Reuse existing ReturnSeries metadata and research dependencies.
**Tech stack:** Python >=3.11, NumPy, existing pandas/scikit-learn research extras.
**Spec:** [Joint design](joint-design.md).
**Execution:** Implement inline, then one independent whole-change review.

## Global constraints

GPL-3.0-only; no copied academic code, no new runtime dependencies, no changes to frozen marginal findings. All observations in each vector must share an input-price interval. Finite slices and sampled optimization must be labeled approximate. No market-data download or new holdout inspection.

## Review focus

- Calendar matches but return intervals differ: reject, never silently align.
- Duplicate distributions exhaust a candidate sample: skip that restart, raise clearly if none is usable.
- Projection counts/atoms/dimensions change: freeze state and reject incompatible prediction.
- Large batches: avoid full projected-training tensors and N-by-N matrices.
- Failed refits or malformed archives: preserve prior fit or reject without loading pickle.

## Tasks

- [x] 1. Geometry and optimizer: create src/wasserstein_regimes/sliced.py and tests/test_sliced.py. Tests first: compare one-dimensional distances to pairwise_distance, prove identical-marginal/opposite-dependence separation, check streamed vs single-batch equality, deterministic fits and malformed archives. Implement fixed unit projections, explicit scales, candidate matrices, iterative medoid updates and streamed full-data objective. Save/load with required fields and finite/dimension checks.
- [x] 2. Temporal alignment: create src/wasserstein_regimes/joint.py and tests/test_joint.py. Tests first: share dates but change preceding prices (reject); insert NaN in one asset (exclude affected windows); change future returns (earlier tensors/availability unchanged); latest component availability must win. Add immutable metadata and explicit symbol ordering.
- [x] 3. Measured experiment and publication: create examples/joint_study.py with independent synthetic train/test samples, marginal/covariance baselines, projection-count/seed sweep and bounded scaling benchmark. Run it, save aggregate JSON, document actual results. Add usage and research links to README/navigation; correct obsolete implementation status. Run full tests, strict documentation build and independent review; open a PR and verify CI.

Verification commands:

```sh
PYTHONPATH=src python -m pytest tests/test_sliced.py tests/test_joint.py -q
PYTHONPATH=src python examples/joint_study.py --output results/joint-synthetic.json
PYTHONPATH=src python -m pytest -q
python -m mkdocs build --strict
```
