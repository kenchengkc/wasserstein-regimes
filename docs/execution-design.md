# Frozen scoring and resumable execution design

Roadmap phase 5. The purpose is reliable reuse and scaling of research computations, not improved regime identification. Phase 4's adverse findings remain binding context.

## Selected approach

Use local, isolated Python subprocesses with a bounded supervisor. Threads alone would share numerical-library state and contaminate memory measurements; a distributed queue would add services before the local recovery contract is established. Linux and macOS are supported for execution via OS advisory locks and process RSS. The NumPy numerical estimators retain their existing portability.

### Frozen joint inference

`freeze-joint` verifies a sealed development artifact and exports either `scaled_joint` or `raw_joint` into a sealed bundle. Its contract records schema version, ordered symbols, window length, provider, adjusted-close/log-return/calendar convention, training end, validation calibration end, novelty threshold, original numerical source hashes and dependency versions. The bundle contains the original model bytes; no fitting occurs. Local bundles contain observed medoids and must not be included in public evidence.

`score-joint` consumes a strict YAML snapshot specification, start/end session dates and explicit UTC-aware as-of time. Ordered symbols, provider and return convention must match the bundle. Snapshot and sidecar hashes/identities are checked. Crop only common history and the scoring endpoint; reject misaligned sessions/return intervals through the existing joint window builder. Keep only complete windows whose raw-price start is after training end and whose latest component availability is no later than as-of. No filling, rescaling or fitting. Write labels, nearest distances and novelty flags plus excluded-window counts, immutable input metadata and bundle identity. Novelty uses a threshold fitted through the calibration end: any score at or before that end must be labeled retrospective calibration reuse, never a point-in-time signal. As-of is an observation-availability filter, not proof of vintage data; snapshots remain revised histories.

### Jobs and identities

`run-jobs` accepts strict named jobs of three types: frozen joint scoring, joint training-prefix/seed refits on existing validation anchors, and synthetic sliced-medoid scaling benchmarks. Refit jobs retain the previous study's K and projections, recompute prefix scales and never evaluate assessment. Benchmark controls specify dimensions and optimizer budgets; they measure engineering behavior only.

Job identity hashes normalized computational parameters, required input file contents, a fixed computational source-file inventory, numerical dependency versions, Python major/minor and requested BLAS thread count. Worker count, timeout, report labels, docs and standalone report-only files do not change identity. Inputs and source identities are rechecked inside the worker and again before completion. Changes cannot silently reuse prior results.

Each job directory has an immutable request, a lock file, append-only attempt directories and an atomic result pointer. Each attempt records status and has its own outputs/log. Success seals all outputs before replacing the pointer; reuse verifies the inventory and pointer digest. Failed/interrupted attempts remain inspectable and are retried only on a subsequent command. A killed worker releases its advisory lock; the next owner marks an unfinished attempt interrupted. A second coordinator waits for the same job lock and then reuses completed work. No two workers publish the same job simultaneously. A coordinator killed without cleanup can leave a child completing; the lock still protects publication.

The supervisor launches at most 1–8 fresh processes, each with 1–8 requested BLAS threads set in the environment before numerical imports and enforced through threadpoolctl. The worker count is independent of the scientific job identity. Explicit timeouts terminate a worker's process group. Ctrl-C terminates this supervisor's active workers. Completed jobs survive either event. The supervisor reports any failures rather than declaring a partial batch successful.

### Memory and artifacts

Every worker records process peak RSS through task completion using `resource.getrusage(RUSAGE_SELF)`, normalized to bytes (Linux KiB, macOS bytes), plus raw units, PID, platform, elapsed time, requested/observed numerical threads. This includes interpreter/import/input memory, not just transient allocations, and is a per-worker peak rather than simultaneous machine-wide RSS. Benchmark rows also record input bytes and fit/score times; no speedup claim without a separate comparison.

Public evidence contains aggregate engineering results, complete specs and fingerprints, never prices, frozen market model arrays or per-window market assignments. Prior studies and numerical engines remain unchanged. GPL-3.0-only; no new dependency.

## Acceptance and limits

Tests cover schema rejection, ordered assets, interval/availability fences, training/calibration distinction, saved-model equivalence, corruption detection, atomic success, interrupted retry, concurrent same-job reuse, process isolation, timeout cleanup, cache identity and platform RSS conversion. Real evidence must show exact saved validation-label reproduction, a second batch entirely reused, bounded subprocesses and measured RSS at increasing synthetic sizes. No market data download or assessment retuning.

Local filesystems only: advisory locks/atomic renames are not a distributed-filesystem guarantee. OS process kills before publication leave attempts rather than resumable numerical iterations. Job-level resumption reruns only incomplete jobs; no checkpoint inside a model fit. Future risk forecasting remains a separate research protocol.

References: [Python resource usage](https://docs.python.org/3/library/resource.html), [subprocess process management](https://docs.python.org/3/library/subprocess.html), [Linux getrusage](https://man7.org/linux/man-pages/man2/getrusage.2.html).
