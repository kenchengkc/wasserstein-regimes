# Frozen scoring and resumable jobs implementation plan

**Goal:** Complete roadmap phase 5 without changing previous scientific results.
**Spec:** [Execution design](execution-design.md).
**Architecture:** Standard-library artifact primitives support frozen joint inference; content-bound task specifications feed isolated workers supervised with bounded concurrency.
**Stack:** Existing Python research extra, Linux/macOS; GPL-3.0-only; no new dependencies.
**Execution:** Inline implementation, tests before behavior, one independent whole-branch review before engineering evidence.

## Review focus

Changed data during execution must not publish success. Missing/corrupt outputs must not count as cached. Process death must not leave an unrecoverable lock. Asset ordering and availability must survive bundle serialization. Historical novelty flags using later calibration must be explicit.

- [ ] **1. Frozen inference and integrity.** Add `artifact_store.py` (canonical hashes, complete inventories, atomic JSON, advisory locks) and `frozen.py` (export/load contract, snapshot validation, score without fitting). Tests: tampering, schema/asset order, exact predictions, availability/training fences, retrospective calibration labels. Preserve existing numerical sources.
- [ ] **2. Isolated resumable jobs.** Add `jobs.py` (strict job schemas, normalized fingerprints, bounded subprocess supervisor), `job_worker.py` (lock/attempt/status/result lifecycle, time/memory/thread metrics), `job_tasks.py` (score/refit/benchmark dispatch). Tests: real process isolation, cache reuse, interrupted retry, lock serialization, timeout cleanup, parameter/input/source identity and report exclusion. Register CLI commands.
- [ ] **3. Review and measured evidence.** Review against design, fix material findings with red/green regressions, export actual historical bundle, reproduce all 690 saved validation labels, execute six benchmarks and four prefix/seed refits with two workers/one BLAS thread. Rerun once to prove all jobs reused. Publish aggregates and measured RSS, update roadmap/docs, pass full suite/strict docs/CI, open PR without merging.

Verification: `PYTHONPATH=src VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m pytest -q`; existing docs environment `python -m mkdocs build --strict`; `git diff --check`. Expected: complete passing suite, verified inventories, clean numerical compatibility, explicit failures instead of partial success claims.
