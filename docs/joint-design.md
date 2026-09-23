# Joint distribution engine design

## Purpose and scope

Implement phases B/C of the existing improvement plan: preserve aligned vector observations within empirical distributions and test dependence changes invisible to marginals. Deliver an optional research engine, synthetic evidence and a scaling budget. GPL-3.0-only, Python >=3.11, NumPy runtime; research examples may use the existing research extras. Existing univariate algorithms, snapshots and published results retain their meaning.

## Representation and geometry

Input is a finite float64 tensor (windows, atoms, assets), equal atom count and fixed asset order. A separate builder accepts a mapping of ReturnSeries with exactly matching session and input-price intervals, rejects mismatches, and drops any whole window containing a missing component. No implicit join, fill or unequal return horizons. Window availability is the maximum over all component inputs. Preserve dates and raw price intervals.

Normalize only by an explicitly supplied positive per-asset scale vector (default ones); the engine never estimates scales from prediction inputs. Callers must fit scales on unique training returns, not duplicated rolling windows. For fixed unit directions theta_r, sort each projected window and retain all atoms. Squared distance is mean over directions and sorted atoms of squared differences. This is exact for the finite projection set, approximates sliced W2 for random directions, and is not full multivariate W2. In one dimension it reduces to exact empirical W2.

## Prototype optimization and memory

SlicedWassersteinKMedoids uses sampled candidate windows, with n_init reproducible samples. Compute only a candidate-by-candidate squared-distance matrix. Alternate assignment and within-cluster medoid selection on candidates, retaining the old medoid on a tie. Select the restart with lowest full-training objective using streamed chunk-to-medoid scoring. Return actual observed joint windows as prototypes and save their original indices. This is an approximate sampled optimizer for squared sliced-W2 medoids, not PAM/SDP or a barycenter solver. Candidate sampling can miss rare states; disclose the budget and sample indices.

For N windows, L atoms, D assets, R projections, M candidates, K prototypes and B scoring batch:
resident input O(NLD), candidate features O(MLR), candidate distances O(M²), scoring workspace O(BLR + BK), saved medoids O(KLD + KLR). Fit stores labels O(N); transform necessarily returns O(NK). No O(N²) matrix when M is fixed. Use direct differences to avoid cancellation in norm-expansion distance formulas.

## API and persistence

- joint_windows(series: mapping, length=63, stride=1) -> JointWindowBatch with samples, symbols, dates, price_start, price_end, available_at.
- SlicedWassersteinKMedoids(n_clusters=2, n_projections=64, candidate_size=128, n_init=3, max_iter=100, chunk_size=32, random_state=42, scales=None, projections=None).
- fit(X), predict(X), transform(X): last returns true distances; inertia_ is sum of squared distances.
- Projections/scales are copied and frozen during fit; prediction validates asset/atom dimensions.
- save(path)/load(path): a versioned NPZ archive without pickle, required keys and finite shape validation; store projections, scales, observed medoids, indices, training labels, configuration and objective. Reject unknown schema, incompatible shape, non-unit directions and nonpositive scales.
- Failed refit leaves the previous fitted model usable: calculate local state, publish only after successful validation/optimization.

## Validation and evidence

Hand-calculated dependence-only windows share exact marginals and differ jointly. Add 1D oracle, row permutation, batch equivalence, frozen state, duplicates/insufficient distinct candidates, invalid shapes/directions/scales, deterministic restarts, reload, gap/interval/availability and future-perturbation checks. Synthetic study uses independent train/test bivariate Gaussian windows with correlations ±0.8; compare marginal and covariance baselines, multiple projection counts/seeds, and report every row. The deterministic control proves the geometric blind spot; random controls estimate finite-sample performance. Performance sample records dimensions, seed, configuration, elapsed fit/score time, environment and traced allocations (not process RSS). Real market joint claims and risk forecasting remain subsequent protocols.
