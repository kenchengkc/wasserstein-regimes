# SPDX-License-Identifier: GPL-3.0-only
"""Lloyd clustering for one-dimensional empirical distributions."""

from __future__ import annotations

import json
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from .transport import _pairwise_sorted_costs, _windows, pairwise_distance


class WassersteinKMeans:
    """Cluster equal-size one-dimensional empirical distributions.

    W2 minimizes squared distances and uses mean quantile barycenters. W1
    minimizes distances and uses coordinate-wise median barycenters.
    """

    _FORMAT = "wasserstein-kmeans"
    _FORMAT_VERSION = 1

    def __init__(
        self,
        metric="w2",
        n_clusters=4,
        n_init=20,
        initialization="kmeans++",
        random_state=42,
        max_iter=300,
        tol=1e-8,
        chunk_size=256,
    ):
        if metric not in ("w1", "w2"):
            raise ValueError("metric must be 'w1' or 'w2'")
        self.n_clusters = self._positive_int(n_clusters, "n_clusters")
        requested_n_init = self._positive_int(n_init, "n_init")
        self.max_iter = self._positive_int(max_iter, "max_iter")
        self.chunk_size = self._positive_int(chunk_size, "chunk_size")
        if isinstance(tol, bool) or not isinstance(tol, Real) or not math.isfinite(tol) or tol < 0:
            raise ValueError("tol must be a finite nonnegative number")
        if random_state is not None and (
            isinstance(random_state, bool)
            or not isinstance(random_state, Integral)
            or random_state < 0
        ):
            raise ValueError("random_state must be an integer or None")

        if isinstance(initialization, str):
            if initialization not in ("kmeans++", "random"):
                raise ValueError("initialization must be 'kmeans++', 'random', or an explicit array")
            stored_initialization = initialization
            effective_n_init = requested_n_init
        else:
            stored_initialization = _windows(initialization, "initialization").copy()
            if stored_initialization.shape[0] != self.n_clusters:
                raise ValueError("explicit initialization must have n_clusters rows")
            stored_initialization = np.sort(stored_initialization, axis=1)
            if len(np.unique(stored_initialization, axis=0)) != self.n_clusters:
                raise ValueError("explicit initialization must contain distinct distributions")
            effective_n_init = 1

        self.metric = metric
        self.distance_metric = metric
        self.objective_power = 1 if metric == "w1" else 2
        self.n_init = effective_n_init
        self.initialization = stored_initialization
        self.random_state = None if random_state is None else int(random_state)
        self.tol = float(tol)

    @staticmethod
    def _positive_int(value, name):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    def _objective_costs(self, samples, centers):
        return _pairwise_sorted_costs(samples, centers, self.objective_power, self.chunk_size)

    def _initial_centers(self, samples, rng):
        if not isinstance(self.initialization, str):
            if self.initialization.shape[1] != samples.shape[1]:
                raise ValueError("explicit initialization atom count must match samples")
            return self.initialization.copy()

        unique = np.unique(samples, axis=0)
        if self.initialization == "random":
            indices = rng.choice(len(unique), self.n_clusters, replace=False)
            return unique[indices].copy()

        chosen = [int(rng.integers(len(unique)))]
        while len(chosen) < self.n_clusters:
            costs = self._objective_costs(unique, unique[chosen]).min(axis=1)
            costs[chosen] = 0.0
            total = float(costs.sum())
            if not math.isfinite(total) or total <= 0:
                remaining = np.setdiff1d(np.arange(len(unique)), chosen, assume_unique=False)
                chosen.append(int(rng.choice(remaining)))
            else:
                chosen.append(int(rng.choice(len(unique), p=costs / total)))
        return unique[chosen].copy()

    def _barycenters(self, samples, labels, previous):
        centers = previous.copy()
        empty = []
        for cluster in range(self.n_clusters):
            members = samples[labels == cluster]
            if len(members) == 0:
                empty.append(cluster)
            elif self.metric == "w1":
                centers[cluster] = np.median(members, axis=0)
            else:
                centers[cluster] = np.mean(members, axis=0)

        if empty:
            occupied = [cluster for cluster in range(self.n_clusters) if cluster not in empty]
            residual = self._objective_costs(samples, centers[occupied]).min(axis=1)
            available = np.ones(len(samples), dtype=bool)
            for cluster in empty:
                candidate_costs = np.where(available, residual, -np.inf)
                index = int(np.argmax(candidate_costs))
                centers[cluster] = samples[index]
                available[index] = False
                residual = np.minimum(
                    residual,
                    self._objective_costs(samples, centers[cluster : cluster + 1])[:, 0],
                )
        return centers

    def _run(self, samples, initial):
        centers = initial.copy()
        previous_labels = None
        initial_costs = self._objective_costs(samples, centers)
        history = [float(initial_costs.min(axis=1).sum())]
        converged = False

        for iteration in range(1, self.max_iter + 1):
            labels = self._objective_costs(samples, centers).argmin(axis=1)
            new_centers = self._barycenters(samples, labels, centers)
            updated_costs = self._objective_costs(samples, new_centers)
            updated_labels = updated_costs.argmin(axis=1)
            objective = float(updated_costs[np.arange(len(samples)), updated_labels].sum())
            history.append(objective)
            shift = float(np.max(np.abs(new_centers - centers)))
            occupied = len(np.unique(updated_labels)) == self.n_clusters
            centers = new_centers
            if occupied and (
                np.array_equal(updated_labels, labels)
                or (previous_labels is not None and np.array_equal(updated_labels, previous_labels))
                or shift <= self.tol
            ):
                converged = True
                break
            previous_labels = updated_labels

        final_costs = self._objective_costs(samples, centers)
        final_labels = final_costs.argmin(axis=1)
        final_objective = float(final_costs[np.arange(len(samples)), final_labels].sum())
        history[-1] = final_objective
        return centers, final_labels, final_objective, tuple(history), iteration, converged

    def fit(self, samples):
        quantiles = np.sort(_windows(samples, "samples"), axis=1)
        unique = np.unique(quantiles, axis=0)
        if len(unique) < self.n_clusters:
            raise ValueError("n_clusters exceeds the number of distinct distributions")
        rng = np.random.default_rng(self.random_state)
        results = [
            self._run(quantiles, self._initial_centers(quantiles, rng))
            for _ in range(self.n_init)
        ]
        centers, labels, inertia, history, n_iter, converged = min(
            results, key=lambda result: result[2]
        )
        self.centers_ = centers.copy()
        self.labels_ = labels.astype(np.int64, copy=True)
        self.inertia_ = float(inertia)
        self.objective_history_ = history
        self.n_iter_ = int(n_iter)
        self.converged_ = bool(converged)
        self.n_features_in_ = quantiles.shape[1]
        self.centers_.setflags(write=False)
        self.labels_.setflags(write=False)
        return self

    def _require_fitted(self):
        if not hasattr(self, "centers_"):
            raise RuntimeError("WassersteinKMeans is not fitted")

    def transform(self, samples):
        self._require_fitted()
        windows = _windows(samples, "samples")
        if windows.shape[1] != self.n_features_in_:
            raise ValueError("sample atom count must match fitted centers")
        return pairwise_distance(
            windows, self.centers_, metric=self.metric, chunk_size=self.chunk_size
        )

    def predict(self, samples):
        return self.transform(samples).argmin(axis=1)

    def fit_predict(self, samples):
        return self.fit(samples).labels_

    @staticmethod
    def _paths(path):
        path = Path(path)
        if path.suffix == ".json":
            return path, path.with_suffix(".npz")
        if path.suffix == ".npz":
            return path.with_suffix(".json"), path
        return path.with_suffix(".json"), path.with_suffix(".npz")

    def save(self, path):
        self._require_fitted()
        metadata_path, arrays_path = self._paths(path)
        metadata = {
            "format": self._FORMAT,
            "format_version": self._FORMAT_VERSION,
            "distance_metric": self.distance_metric,
            "objective_power": self.objective_power,
            "n_clusters": self.n_clusters,
            "n_init": self.n_init,
            "initialization": "explicit" if not isinstance(self.initialization, str) else self.initialization,
            "random_state": self.random_state,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "chunk_size": self.chunk_size,
            "n_features_in": self.n_features_in_,
            "n_samples": len(self.labels_),
            "inertia": self.inertia_,
            "converged": self.converged_,
            "n_iter": self.n_iter_,
        }
        np.savez_compressed(
            arrays_path,
            centers=self.centers_,
            labels=self.labels_,
            objective_history=np.asarray(self.objective_history_, dtype=np.float64),
        )
        metadata_path.write_text(
            json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path):
        metadata_path, arrays_path = cls._paths(path)
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid model metadata") from error
        required = {
            "format", "format_version", "distance_metric", "objective_power",
            "n_clusters", "n_init", "initialization", "random_state", "max_iter",
            "tol", "chunk_size", "n_features_in", "n_samples", "inertia",
            "converged", "n_iter",
        }
        if not isinstance(metadata, dict) or set(metadata) != required:
            raise ValueError("invalid model metadata")
        metric = metadata.get("distance_metric")
        expected_power = 1 if metric == "w1" else 2 if metric == "w2" else None
        if (
            metadata.get("format") != cls._FORMAT
            or metadata.get("format_version") != cls._FORMAT_VERSION
            or metadata.get("objective_power") != expected_power
            or not isinstance(metadata.get("converged"), bool)
        ):
            raise ValueError("invalid model metadata")
        initialization = metadata["initialization"]
        if initialization == "explicit":
            initialization = "kmeans++"
        try:
            model = cls(
                metric=metric,
                n_clusters=metadata["n_clusters"],
                n_init=metadata["n_init"],
                initialization=initialization,
                random_state=metadata["random_state"],
                max_iter=metadata["max_iter"],
                tol=metadata["tol"],
                chunk_size=metadata["chunk_size"],
            )
            with np.load(arrays_path, allow_pickle=False) as archive:
                if set(archive.files) != {"centers", "labels", "objective_history"}:
                    raise ValueError("invalid model arrays")
                centers = archive["centers"].copy()
                labels = archive["labels"].copy()
                history = archive["objective_history"].copy()
        except (OSError, TypeError, ValueError) as error:
            raise ValueError("invalid model arrays or metadata") from error

        n_features = metadata["n_features_in"]
        n_samples = metadata["n_samples"]
        inertia = metadata["inertia"]
        n_iter = metadata["n_iter"]
        valid_arrays = (
            centers.shape == (model.n_clusters, n_features)
            and centers.ndim == 2
            and np.issubdtype(centers.dtype, np.number)
            and np.isfinite(centers).all()
            and np.all(np.diff(centers, axis=1) >= 0)
            and labels.shape == (n_samples,)
            and np.issubdtype(labels.dtype, np.integer)
            and np.all((labels >= 0) & (labels < model.n_clusters))
            and history.ndim == 1
            and len(history) >= 1
            and np.isfinite(history).all()
        )
        valid_scalars = (
            isinstance(n_features, Integral) and not isinstance(n_features, bool) and n_features > 0
            and isinstance(n_samples, Integral) and not isinstance(n_samples, bool) and n_samples > 0
            and isinstance(n_iter, Integral) and not isinstance(n_iter, bool) and 1 <= n_iter <= model.max_iter
            and isinstance(inertia, Real) and not isinstance(inertia, bool) and math.isfinite(inertia) and inertia >= 0
        )
        if not valid_arrays or not valid_scalars or not np.isclose(history[-1], inertia, rtol=1e-12, atol=1e-12):
            raise ValueError("invalid model centers, labels, or metadata")

        model.centers_ = np.asarray(centers, dtype=np.float64).copy()
        model.labels_ = np.asarray(labels, dtype=np.int64).copy()
        model.objective_history_ = tuple(float(value) for value in history)
        model.n_features_in_ = int(n_features)
        model.inertia_ = float(inertia)
        model.converged_ = metadata["converged"]
        model.n_iter_ = int(n_iter)
        model.centers_.setflags(write=False)
        model.labels_.setflags(write=False)
        return model
