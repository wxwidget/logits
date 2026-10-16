"""Thin interfaces to the constrained C++ least-squares solvers."""
import numpy as np

from . import _core, _dataset, _work
from .analysis import _targets, _weights, ols_summary, regression_metrics


def _endpoint(value, size, default):
    if value is None:
        return np.full(size, default)
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(size, float(array))
    if array.shape != (size,):
        raise ValueError("coefficient bound must be scalar or one value per feature")
    return array.copy()


class LinearRegression:
    """Weighted mean squared loss / 2 + l1*|w|_1 + l2*||w||²/2.

    Intercept is unpenalized. Bounds apply in the coordinates supplied by the
    caller; the estimator does not silently scale features. Prediction uses
    the owned C++ result; fitted arrays are inspection copies.
    """
    def __init__(self, *, solver="auto", l1=0., l2=0., fit_intercept=True,
                 max_iter=1000, tolerance=1e-7, learning_rate=1., positive=False,
                 bounds=None, fixed_coefficients=None, intercept_bounds=None):
        self.options = _core.LinearOptions()
        for key in ("solver", "l1", "l2", "fit_intercept", "max_iter", "tolerance",
                    "learning_rate", "positive"):
            setattr(self.options, key, locals()[key])
        self.bounds = bounds
        self.fixed_coefficients = dict(fixed_coefficients or {})
        self.intercept_bounds = intercept_bounds

    def fit(self, X, y, sample_weight=None):
        y = _targets(y)
        # Core conversion validates feature storage before any computation.
        data = _dataset(X, y, binary_labels=False)
        features = X.shape[1] if hasattr(X, "shape") else np.asarray(X).shape[1]
        options = _core.LinearOptions(self.options)
        lower, upper = (None, None) if self.bounds is None else self.bounds
        lower = _endpoint(lower, features, -np.inf)
        upper = _endpoint(upper, features, np.inf)
        if options.positive:
            lower = np.maximum(lower, 0.)
        for index, value in self.fixed_coefficients.items():
            if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or not 0 <= index < features:
                raise ValueError("fixed coefficient index must be a valid integer feature index")
            if not np.isfinite(value) or not lower[index] <= value <= upper[index]:
                raise ValueError("fixed coefficient must be finite and satisfy all bounds")
            lower[index] = upper[index] = value
        options.lower_bounds = lower.tolist()
        options.upper_bounds = upper.tolist()
        if self.intercept_bounds is not None:
            options.intercept_lower, options.intercept_upper = self.intercept_bounds
        weights = _weights(sample_weight, len(y)) if sample_weight is not None else []
        result = _core.fit_linear(data, options, weights)
        # Publish fitted state only after success, preserving a previous model on errors.
        self._result = result
        self._fit_options = options
        self._unconstrained = (not np.isfinite(lower).any() and not np.isfinite(upper).any()
                               and (not options.fit_intercept or
                                    (options.intercept_lower == -np.inf and options.intercept_upper == np.inf)))
        self.n_features_in_ = features
        self.n_samples_ = len(y)
        self.coef_ = np.asarray(result.weights)
        self.intercept_ = result.intercept
        self.converged_ = result.converged
        self.status_ = result.status
        self.solver_ = result.solver
        self.work_ = _work(result.work)
        self.history_ = [dict(iteration=h.iteration, loss=h.loss,
                              stationarity_norm=h.stationarity_norm, seconds=h.seconds,
                              **_work(h.work)) for h in result.history]
        return self

    def predict(self, X):
        if not hasattr(self, "_result"):
            raise RuntimeError("fit must be called before prediction")
        return np.asarray(_core.predict_linear(_dataset(X), self._result))

    def residuals(self, X, y):
        y = _targets(y)
        prediction = self.predict(X)
        if y.shape != prediction.shape:
            raise ValueError("target count does not match predictions")
        return y - prediction

    def evaluate(self, X, y, sample_weight=None):
        return regression_metrics(y, self.predict(X), sample_weight=sample_weight)

    def score(self, X, y, sample_weight=None):
        return self.evaluate(X, y, sample_weight)["r2"]

    def summary(self, X, y, *, sample_weight=None, confidence=.95,
                covariance="classical", max_features=512):
        if not hasattr(self, "_result"):
            raise RuntimeError("fit must be called before summary")
        o = self._fit_options
        if not self.converged_ or o.l1 != 0 or o.l2 != 0 or not self._unconstrained:
            raise ValueError("inference requires converged unregularized, unconstrained least squares")
        return ols_summary(X, y, self._result.weights, self._result.intercept,
                           fit_intercept=o.fit_intercept, sample_weight=sample_weight,
                           confidence=confidence, covariance=covariance,
                           max_features=max_features, fit_tolerance=o.tolerance)


class Ridge(LinearRegression):
    """Mean-loss Ridge with l2=alpha (sklearn Ridge uses sum-loss scaling)."""
    def __init__(self, *, alpha=1., **kwargs):
        super().__init__(l1=0., l2=alpha, **kwargs)


class Lasso(LinearRegression):
    """Lasso with l1=alpha."""
    def __init__(self, *, alpha=1., **kwargs):
        super().__init__(l1=alpha, l2=0., **kwargs)


class ElasticNet(LinearRegression):
    """l1=alpha*l1_ratio; l2=alpha*(1-l1_ratio)."""
    def __init__(self, *, alpha=1., l1_ratio=.5, **kwargs):
        if not np.isfinite(l1_ratio) or not 0 <= l1_ratio <= 1:
            raise ValueError("l1_ratio must be finite and in [0, 1]")
        super().__init__(l1=alpha * l1_ratio, l2=alpha * (1 - l1_ratio), **kwargs)
