# SPDX-License-Identifier: GPL-3.0-only
"""Causal comparison baselines for distributional regime experiments."""

import warnings
from numbers import Integral

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


_FEATURE_KINDS = {"volatility", "mean_vol", "moments", "rich", "gmm"}
_QUANTILES = np.array([.01, .05, .1, .25, .5, .75, .9, .95, .99])


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _samples(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or 0 in values.shape or not np.isfinite(values).all():
        raise ValueError("samples must be a nonempty finite 2D array")
    return values


def _moment_features(samples):
    means = samples.mean(axis=1)
    standard_deviations = samples.std(axis=1, ddof=0)
    constant = np.all(samples == samples[:, :1], axis=1)
    standard_deviations[constant] = 0.0
    centered = samples - means[:, None]
    standardized = np.divide(
        centered,
        standard_deviations[:, None],
        out=np.zeros_like(centered),
        where=standard_deviations[:, None] > 0,
    )
    skewness = np.mean(standardized ** 3, axis=1)
    excess_kurtosis = np.mean(standardized ** 4, axis=1) - 3.0
    skewness[constant] = 0.0
    excess_kurtosis[constant] = 0.0
    return np.column_stack([means, standard_deviations, skewness, excess_kurtosis])


def _maximum_drawdown(samples):
    cumulative_log_returns = np.column_stack([
        np.zeros(len(samples)), np.cumsum(samples, axis=1),
    ])
    running_peaks = np.maximum.accumulate(cumulative_log_returns, axis=1)
    drawdowns = 1.0 - np.exp(cumulative_log_returns - running_peaks)
    return drawdowns.max(axis=1)


def feature_matrix(samples, kind="moments"):
    """Return deterministic window features without fitting any data transform."""
    samples = _samples(samples)
    if kind not in _FEATURE_KINDS:
        raise ValueError(f"unknown feature kind: {kind!r}")
    moments = _moment_features(samples)
    if kind == "volatility":
        return moments[:, 1:2]
    if kind == "mean_vol":
        return moments[:, :2]
    if kind in {"moments", "gmm"}:
        return moments
    quantiles = np.quantile(samples, _QUANTILES, axis=1).T
    downside = np.sqrt(np.mean(np.minimum(samples, 0.0) ** 2, axis=1))
    return np.column_stack([
        moments, quantiles, downside, _maximum_drawdown(samples),
    ])


def _approximately_distinct_rows(values, limit):
    representatives = []
    for row in values:
        if not any(np.allclose(row, other, rtol=1e-12, atol=1e-12) for other in representatives):
            representatives.append(row)
            if len(representatives) == limit:
                break
    return np.asarray(representatives)


class FeatureBaseline:
    """K-means or Gaussian-mixture clustering on training-scaled window features."""

    def __init__(self, kind="moments", n_clusters=3, random_state=42, n_init=20):
        if kind not in _FEATURE_KINDS:
            raise ValueError(f"unknown feature kind: {kind!r}")
        _positive_int(n_clusters, "n_clusters")
        _positive_int(n_init, "n_init")
        self.kind = kind
        self.n_clusters = int(n_clusters)
        self.random_state = random_state
        self.n_init = int(n_init)

    def fit(self, samples):
        samples = _samples(samples)
        if len(samples) < self.n_clusters:
            raise ValueError("n_clusters cannot exceed the number of samples")
        raw_features = feature_matrix(samples, self.kind)
        self.scaler_ = StandardScaler().fit(raw_features)
        scaled_features = self.scaler_.transform(raw_features)
        distinct = _approximately_distinct_rows(raw_features, self.n_clusters)
        self.n_effective_clusters_ = min(self.n_clusters, len(distinct))

        if self.kind == "gmm":
            self.model_ = GaussianMixture(
                n_components=self.n_effective_clusters_,
                n_init=self.n_init,
                random_state=self.random_state,
                reg_covar=1e-6,
            ).fit(scaled_features)
        else:
            self.model_ = KMeans(
                n_clusters=self.n_effective_clusters_,
                n_init=self.n_init,
                random_state=self.random_state,
            ).fit(scaled_features)

        labels = self.model_.predict(scaled_features)
        occupied = np.unique(labels)
        if len(occupied) != self.n_effective_clusters_:
            raise ValueError("fitted feature model produced an empty cluster")
        remap = {old: new for new, old in enumerate(occupied)}
        self._label_map_ = remap
        self.labels_ = np.array([remap[label] for label in labels], dtype=int)
        sorted_samples = np.sort(samples, axis=1)
        self.centers_ = np.stack([
            sorted_samples[self.labels_ == cluster].mean(axis=0)
            for cluster in range(self.n_effective_clusters_)
        ])
        return self

    def _scaled_features(self, samples):
        if not hasattr(self, "model_"):
            raise RuntimeError("FeatureBaseline must be fitted before use")
        return self.scaler_.transform(feature_matrix(samples, self.kind))

    def predict(self, samples):
        scaled = self._scaled_features(samples)
        labels = self.model_.predict(scaled)
        try:
            return np.array([self._label_map_[label] for label in labels], dtype=int)
        except KeyError as error:
            raise ValueError("prediction selected a component without a training prototype") from error

    def transform(self, samples):
        scaled = self._scaled_features(samples)
        if self.kind == "gmm":
            return 1.0 - self.model_.predict_proba(scaled)[:, sorted(self._label_map_)]
        return self.model_.transform(scaled)[:, sorted(self._label_map_)]


def _returns(values, *, minimum=1):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < minimum or not np.isfinite(values).all():
        raise ValueError(f"returns must be a finite 1D array with at least {minimum} values")
    return values


def _normalized_probabilities(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"{name} must contain finite nonnegative probabilities")
    total = values.sum()
    if total <= 0:
        raise ValueError(f"{name} probabilities must have positive total mass")
    return values / total


def _logsumexp(values, axis=None):
    maximum = np.max(values, axis=axis, keepdims=True)
    with np.errstate(under="ignore"):
        result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
    if axis is not None:
        result = np.squeeze(result, axis=axis)
    return result


class CausalGaussianHMM:
    """Univariate Gaussian HMM with explicit causal forward filtering."""

    _PROBABILITY_FLOOR = 1e-12
    _SCALE_FLOOR = 1e-12
    _COVARIANCE_FLOOR = 1e-8

    def __init__(self, n_clusters=3, random_state=42, n_init=5, max_iter=200):
        for name, value in (
            ("n_clusters", n_clusters), ("n_init", n_init), ("max_iter", max_iter),
        ):
            _positive_int(value, name)
        self.n_clusters = int(n_clusters)
        self.random_state = random_state
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)

    @classmethod
    def _floor_probability_vector(cls, values):
        values = np.maximum(np.asarray(values, dtype=np.float64), cls._PROBABILITY_FLOOR)
        return values / values.sum()

    @classmethod
    def _floor_transition_matrix(cls, values):
        values = np.maximum(np.asarray(values, dtype=np.float64), cls._PROBABILITY_FLOOR)
        return values / values.sum(axis=1, keepdims=True)

    def fit(self, returns1d):
        free_parameters = self.n_clusters ** 2 + 2 * self.n_clusters - 1
        returns = _returns(returns1d, minimum=free_parameters)
        if len(np.unique(returns)) < self.n_clusters:
            raise ValueError("returns must contain at least n_clusters distinct values")
        self.train_mean_ = float(returns.mean())
        self.train_std_ = float(max(returns.std(ddof=0), self._SCALE_FLOOR))
        observations = ((returns - self.train_mean_) / self.train_std_)[:, None]

        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as error:
            raise ImportError("CausalGaussianHMM requires the optional hmmlearn dependency") from error

        seed_generator = np.random.default_rng(self.random_state)
        candidates = []
        diagnostics = []
        for _ in range(self.n_init):
            seed = int(seed_generator.integers(0, np.iinfo(np.int32).max))
            diagnostic = {"seed": seed}
            try:
                candidate = GaussianHMM(
                    n_components=self.n_clusters,
                    covariance_type="diag",
                    min_covar=self._COVARIANCE_FLOOR,
                    n_iter=self.max_iter,
                    random_state=seed,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    candidate.fit(observations)
                    loglikelihood = float(candidate.score(observations))
                n_iter = int(candidate.monitor_.iter)
                reached_limit = n_iter >= self.max_iter
                history = [float(value) for value in candidate.monitor_.history]
                likelihood_decreased = any(
                    current < previous - 1e-6
                    for previous, current in zip(history, history[1:])
                )
                converged = (
                    bool(candidate.monitor_.converged)
                    and not reached_limit
                    and not likelihood_decreased
                )
                parameters = (
                    candidate.startprob_, candidate.transmat_,
                    candidate.means_, candidate.covars_,
                )
                valid = np.isfinite(loglikelihood) and all(
                    np.isfinite(parameter).all() for parameter in parameters
                )
                diagnostic.update({
                    "valid": bool(valid),
                    "converged": converged,
                    "n_iter": n_iter,
                    "reached_iteration_limit": reached_limit,
                    "training_loglikelihood": loglikelihood,
                    "loglikelihood_history": history,
                    "likelihood_decreased": likelihood_decreased,
                    "warnings": [str(item.message) for item in caught],
                })
                if valid:
                    candidates.append((candidate, loglikelihood, diagnostic))
            except (ValueError, FloatingPointError) as error:
                diagnostic.update({"valid": False, "error": str(error), "warnings": []})
            diagnostics.append(diagnostic)

        if not candidates:
            raise RuntimeError("all Gaussian HMM initializations failed")
        converged_candidates = [item for item in candidates if item[2]["converged"]]
        best, loglikelihood, selected = max(
            converged_candidates or candidates, key=lambda item: item[1],
        )
        self.startprob_ = self._floor_probability_vector(best.startprob_)
        self.transmat_ = self._floor_transition_matrix(best.transmat_)
        self.means_ = np.asarray(best.means_, dtype=np.float64).reshape(self.n_clusters)
        self.covars_ = np.maximum(
            np.asarray(best.covars_, dtype=np.float64).reshape(self.n_clusters, -1)[:, 0],
            self._COVARIANCE_FLOOR,
        )
        self.converged_ = bool(selected["converged"])
        self.n_iter_ = int(selected["n_iter"])
        self.reached_iteration_limit_ = bool(selected["reached_iteration_limit"])
        self.training_loglikelihood_ = float(loglikelihood)
        self.fit_diagnostics_ = diagnostics
        self.terminal_proba_ = self.filter_proba(returns)[-1]
        return self

    def _check_fitted(self):
        required = ("train_mean_", "train_std_", "startprob_", "transmat_", "means_", "covars_")
        if not all(hasattr(self, name) for name in required):
            raise RuntimeError("CausalGaussianHMM must be fitted before use")

    def filter_proba(self, returns1d, initial=None):
        """Return causal state posteriors; explicit ``initial`` is the prior terminal posterior."""
        self._check_fitted()
        returns = _returns(returns1d)
        scaled = (returns - self.train_mean_) / self.train_std_
        if not np.isfinite(scaled).all():
            raise ValueError("scaled returns must be finite")
        if initial is None:
            prior = self._floor_probability_vector(self.startprob_)
        else:
            previous = _normalized_probabilities(initial, "initial")
            if previous.shape != (self.n_clusters,):
                raise ValueError(f"initial must have shape ({self.n_clusters},)")
            prior = previous @ self.transmat_

        log_transition = np.log(self._floor_transition_matrix(self.transmat_))
        log_probabilities = np.empty((len(scaled), self.n_clusters))
        log_prior = np.log(self._floor_probability_vector(prior))
        log_normalizer = .5 * np.log(2.0 * np.pi * self.covars_)
        for index, value in enumerate(scaled):
            with np.errstate(over="ignore", invalid="ignore"):
                emission = -log_normalizer - .5 * (value - self.means_) ** 2 / self.covars_
            log_posterior = log_prior + emission
            normalization = _logsumexp(log_posterior)
            if not np.isfinite(normalization):
                raise ValueError("returns are too large for stable Gaussian evaluation")
            log_posterior -= normalization
            log_probabilities[index] = log_posterior
            log_prior = _logsumexp(log_posterior[:, None] + log_transition, axis=0)
        return np.exp(log_probabilities)

    def predict_filtered(self, returns1d, initial=None):
        return self.filter_proba(returns1d, initial=initial).argmax(axis=1)

    def to_dict(self):
        self._check_fitted()
        return {
            "n_clusters": self.n_clusters,
            "mu": self.means_.tolist(),
            "cov": self.covars_.tolist(),
            "trans": self.transmat_.tolist(),
            "start": self.startprob_.tolist(),
            "train_mean": float(self.train_mean_),
            "train_std": float(self.train_std_),
            "terminal_proba": getattr(self, "terminal_proba_", np.array([])).tolist(),
            "converged": bool(getattr(self, "converged_", False)),
            "n_iter": int(getattr(self, "n_iter_", 0)),
            "reached_iteration_limit": bool(getattr(self, "reached_iteration_limit_", False)),
            "training_loglikelihood": float(getattr(self, "training_loglikelihood_", np.nan)),
            "fit_diagnostics": self.fit_diagnostics_,
        }
