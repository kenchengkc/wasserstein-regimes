# SPDX-License-Identifier: GPL-3.0-only
"""Finite-projection sliced W2 and sampled, squared-distance medoid clustering.

No multivariate barycenters are constructed. Candidate optimization is approximate;
a finite set of projections defines a pseudometric, not full multivariate W2.
"""
from __future__ import annotations

import json
from numbers import Integral
from pathlib import Path

import numpy as np

from .transport import _pairwise_sorted_costs, _positive_int


def _joint(values):
    if np.iscomplexobj(values):
        raise ValueError("samples must be real")
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 3 or 0 in x.shape or not np.isfinite(x).all():
        raise ValueError("samples must be finite nonempty (windows, atoms, assets)")
    return x


def _geometry(dimension, projections, scales):
    directions = np.array(projections, dtype=np.float64, copy=True)
    scale = np.ones(dimension) if scales is None else np.array(scales, dtype=np.float64, copy=True)
    if (directions.ndim != 2 or len(directions) == 0 or directions.shape[1] != dimension
            or not np.isfinite(directions).all()
            or not np.allclose(np.linalg.norm(directions, axis=1), 1., rtol=1e-12, atol=1e-12)):
        raise ValueError("projections must be finite unit directions with one column per asset")
    if scale.shape != (dimension,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("scales must be finite positive values with one entry per asset")
    return directions, scale


def _project(x, directions, scales):
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        q = np.sort((x / scales) @ directions.T, axis=1)
    if not np.isfinite(q).all():
        raise ValueError("scaled projections overflow")
    return q.reshape(len(x), -1)


def _costs(x, y, chunk_size):
    # Already projected/sorted per direction: NEVER sort the flattened vectors.
    with np.errstate(over='ignore', invalid='ignore'):
        out = _pairwise_sorted_costs(x, y, 2, chunk_size)
    if not np.isfinite(out).all():
        raise ValueError("squared sliced distances overflow")
    return out


def sliced_w2(samples, centers, projections, *, scales=None, chunk_size=32):
    """True distances for equally weighted windows and explicit unit directions.

    Uses only block-sized projected samples; output itself has shape (N, K).
    Explicit scales must come from historical training data, if used.
    """
    x, y = _joint(samples), _joint(centers)
    if x.shape[1:] != y.shape[1:]:
        raise ValueError("samples and centers must have identical atom and asset dimensions")
    size = _positive_int(chunk_size, "chunk_size")
    directions, scale = _geometry(x.shape[2], projections, scales)
    out = np.empty((len(x), len(y)))
    for j in range(0, len(y), size):
        qy = _project(y[j:j+size], directions, scale)
        for i in range(0, len(x), size):
            qx = _project(x[i:i+size], directions, scale)
            out[i:i+size, j:j+size] = np.sqrt(_costs(qx, qy, size))
    return out


def _assign(x, centers, directions, scales, size):
    labels = np.empty(len(x), dtype=np.int64)
    objective = 0.
    for start in range(0, len(x), size):
        q = _project(x[start:start+size], directions, scales)
        costs = _costs(q, centers, size)
        chosen = costs.argmin(axis=1)
        labels[start:start+len(q)] = chosen
        objective += float(costs[np.arange(len(q)), chosen].sum())
    if not np.isfinite(objective):
        raise ValueError("objective overflow")
    return labels, objective


class SlicedWassersteinKMedoids:
    """Sample candidate sets, optimize medoids there, choose by full-data loss.

    Complexity depends quadratically on candidate_size, not training window count.
    Rare distributions may be absent from candidates. Custom projections override
    n_projections. Callers must preserve asset ordering at inference.
    """

    def __init__(self, *, n_clusters=2, n_projections=64, candidate_size=128,
                 n_init=3, max_iter=100, chunk_size=32, random_state=42,
                 scales=None, projections=None):
        for name, value in dict(n_clusters=n_clusters, n_projections=n_projections,
                                candidate_size=candidate_size, n_init=n_init,
                                max_iter=max_iter, chunk_size=chunk_size).items():
            setattr(self, name, _positive_int(value, name))
        if candidate_size < n_clusters:
            raise ValueError("candidate_size must be at least n_clusters")
        if isinstance(random_state, bool) or not isinstance(random_state, Integral) or random_state < 0:
            raise ValueError("random_state must be a nonnegative integer")
        self.random_state = int(random_state)
        self.scales = None if scales is None else np.array(scales, dtype=np.float64, copy=True)
        self.projections = None if projections is None else np.array(projections, dtype=np.float64, copy=True)

    def fit(self, samples):
        x = _joint(samples)
        if len(x) < self.n_clusters:
            raise ValueError("not enough distinct candidate distributions")
        rng = np.random.default_rng(self.random_state)
        directions = self.projections
        if directions is None:
            directions = rng.normal(size=(self.n_projections, x.shape[2]))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        directions, scales = _geometry(x.shape[2], directions, self.scales)
        best, runs = None, []
        for _ in range(self.n_init):
            indices = np.sort(rng.choice(len(x), size=min(len(x), self.candidate_size), replace=False))
            features = _project(x[indices], directions, scales)
            _, unique = np.unique(features, axis=0, return_index=True)
            if len(unique) < self.n_clusters:
                runs.append(dict(candidate_indices=indices.tolist(), skipped="insufficient distinct projections"))
                continue
            matrix = _costs(features, features, self.chunk_size)
            medoids = rng.choice(np.sort(unique), size=self.n_clusters, replace=False)
            history = [float(matrix[:, medoids].min(axis=1).sum())]
            converged = False
            for iteration in range(1, self.max_iter + 1):
                labels = matrix[:, medoids].argmin(axis=1)
                updated = medoids.copy()
                for cluster in range(self.n_clusters):
                    members = np.flatnonzero(labels == cluster)
                    # Distinct projected medoids each own at least their own point.
                    sums = matrix[np.ix_(members, members)].sum(axis=0)
                    choice = members[sums.argmin()]
                    old_cost = matrix[members, medoids[cluster]].sum()
                    if sums.min() < old_cost:
                        updated[cluster] = choice
                loss = float(matrix[:, updated].min(axis=1).sum())
                history.append(loss)
                if np.array_equal(updated, medoids):
                    converged = True
                    break
                medoids = updated
            full_labels, objective = _assign(x, features[medoids], directions, scales, self.chunk_size)
            runs.append(dict(candidate_indices=indices.tolist(), objective_history=history,
                             full_objective=objective, converged=converged, n_iter=iteration))
            if best is None or objective < best[0]:
                best = (objective, indices[medoids].copy(), full_labels, iteration, converged)
        if best is None:
            raise ValueError("not enough distinct candidate distributions; increase candidate_size")
        objective, indices, labels, iteration, converged = best
        # Commit state only after a successful fit.
        state = dict(medoids_=x[indices].copy(), medoid_indices_=indices, labels_=labels,
                     projections_=directions, scales_=scales)
        for value in state.values():
            value.setflags(write=False)
        self.__dict__.update(state)
        self.inertia_, self.n_iter_, self.converged_ = objective, iteration, converged
        self.candidate_runs_ = runs
        return self

    def _check(self, samples):
        if not hasattr(self, "medoids_"):
            raise ValueError("model must be fitted")
        x = _joint(samples)
        if x.shape[1:] != self.medoids_.shape[1:]:
            raise ValueError("prediction atom/asset dimensions differ from fitted model")
        return x

    def transform(self, samples):
        x = self._check(samples)
        return sliced_w2(x, self.medoids_, self.projections_, scales=self.scales_, chunk_size=self.chunk_size)

    def predict(self, samples):
        x = self._check(samples)
        centers = _project(self.medoids_, self.projections_, self.scales_)
        return _assign(x, centers, self.projections_, self.scales_, self.chunk_size)[0]

    def save(self, path):
        self._check(self.medoids_)
        params = {name:getattr(self, name) for name in (
            "n_clusters", "n_projections", "candidate_size", "n_init", "max_iter", "chunk_size", "random_state")}
        metadata = dict(version=1, params=params, inertia=self.inertia_, n_iter=self.n_iter_,
                        converged=self.converged_, candidate_runs=self.candidate_runs_)
        # A single archive cannot mix JSON from one model with arrays from another.
        with Path(path).open("wb") as handle:
            np.savez_compressed(handle, metadata=json.dumps(metadata), medoids=self.medoids_,
                                indices=self.medoid_indices_, labels=self.labels_,
                                projections=self.projections_, scales=self.scales_)

    @classmethod
    def load(cls, path):
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"metadata", "medoids", "indices", "labels", "projections", "scales"}:
                    raise ValueError("invalid archive keys")
                metadata = json.loads(str(archive["metadata"].item()))
                if set(metadata) != {"version", "params", "inertia", "n_iter", "converged", "candidate_runs"} or metadata["version"] != 1:
                    raise ValueError("invalid archive metadata")
                model = cls(**metadata["params"], scales=archive["scales"], projections=archive["projections"])
                medoids = _joint(archive["medoids"]).copy()
                directions, scales = _geometry(medoids.shape[2], archive["projections"], archive["scales"])
                indices, labels = archive["indices"].copy(), archive["labels"].copy()
            if (len(medoids) != model.n_clusters or indices.shape != (model.n_clusters,)
                    or labels.ndim != 1 or not len(labels)
                    or not np.issubdtype(indices.dtype, np.integer)
                    or not np.issubdtype(labels.dtype, np.integer)
                    or len(np.unique(indices)) != len(indices)
                    or np.any(indices < 0) or np.any(indices >= len(labels))
                    or np.any(labels < 0) or np.any(labels >= model.n_clusters)
                    or not isinstance(metadata["inertia"], (int, float))
                    or not np.isfinite(metadata["inertia"]) or metadata["inertia"] < 0
                    or not isinstance(metadata["n_iter"], int) or isinstance(metadata["n_iter"], bool)
                    or not 1 <= metadata["n_iter"] <= model.max_iter
                    or not isinstance(metadata["converged"], bool)
                    or not isinstance(metadata["candidate_runs"], list)):
                raise ValueError("invalid model state")
            if len(np.unique(_project(medoids, directions, scales), axis=0)) != model.n_clusters:
                raise ValueError("medoids must be distinct under saved projections")
            for value in (medoids, directions, scales, indices, labels):
                value.setflags(write=False)
            model.medoids_, model.medoid_indices_, model.labels_ = medoids, indices, labels
            model.projections_, model.scales_ = directions, scales
            model.inertia_, model.n_iter_ = metadata["inertia"], metadata["n_iter"]
            model.converged_, model.candidate_runs_ = metadata["converged"], metadata["candidate_runs"]
            return model
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            raise ValueError("invalid sliced model archive") from error
