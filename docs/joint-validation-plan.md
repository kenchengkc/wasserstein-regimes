# Joint robustness implementation plan

> Execute inline with the executing-plans workflow and one independent whole-branch review.

**Goal:** Deliver roadmap phase 4 as a reproducible, auditable stress study.
**Architecture:** Pure array resampling/control functions feed an isolated research runner that loads the frozen development reference, scores common anchors and seals aggregate artifacts.
**Tech stack:** Existing Python/NumPy/SciPy/scikit-learn research extras; no new dependencies.
**Spec:** [Frozen protocol](joint-validation-protocol.md).

## Global constraints

GPL-3.0-only. No assessment evaluation, raw-data publication, silent retuning or predictive claims. Preserve the numerical engine and prior evidence. Inline execution is carried forward from the preceding approved phases.

## Review focus

Shared resampling indices preserve cross-asset dependence; bootstrap seams must be disclosed. Synthetic split windows cannot overlap returns. Rare-state mapping must not use test truth. Parent artifacts and cache identity must reject changed provenance. Collapsed clusters must stay visible rather than yielding misleading stability summaries.

## Tasks

- [x] **1. Primitives and controls.** Create `robustness.py` and `tests/test_robustness.py`. Interfaces: `stationary_indices(n, mean_block, rng) -> (indices, seams)`; `return_windows(vectors, length, stride=1)`; `contaminate(vectors, fraction, magnitude, rng)`; `control_stream(kind, seed)`; `partition_diagnostics(labels,k,reference=None)`. Write tests of synchronized pairs, geometric continuation, nonfinite inputs, disjoint pure-window masks, fixed marginal/correlation parameters, array immutability and collapse flags. Observe failure, implement, run focused tests, commit.
- [x] **2. Research execution.** Create `joint_validation.py`, `configs/joint_validation.yaml`, `tests/test_joint_validation.py`; register `joint-validate` CLI. Consume primitives and existing sealed development models. Produce `run_joint_validation(config_path, development, output_root='artifacts') -> report Path`. Cover parent/source/dependency/config identity, return-only resampling, recomputed scales, common anchors, strict configuration, deterministic seeds and cached artifact tampering. Add synthetic recovery/censoring tests with constructed labels. Observe failure, implement, run full suite, commit before scientific execution.
- [ ] **3. Review and evidence.** Independent code/protocol review; fix material findings with regression tests. Run the frozen command once, verify checksums and result counts, publish aggregate JSON/HTML and measured narrative. Update roadmap/navigation/README; strict documentation build, full suite and CI. Open a dedicated pull request without merging.

Verification commands: `PYTHONPATH=src .venv/bin/python -m pytest -q`; existing docs environment `python -m mkdocs build --strict`; `git diff --check`. Expected: all tests and strict build pass. Research failures are retained and explained rather than silently omitted.
