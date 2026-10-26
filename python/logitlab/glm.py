"""Canonical generalized linear models; optimization and deviance run in C++."""
import numpy as np

from . import _core, _dataset, _work
from .analysis import _targets, _weights


def _offset(offset, exposure, rows, family):
    if offset is None and exposure is None:
        return []
    result = np.zeros(rows) if offset is None else np.asarray(offset, dtype=float).copy()
    if result.shape != (rows,) or not np.isfinite(result).all():
        raise ValueError("offset must be finite with one value per row")
    if exposure is not None:
        if family != "poisson":
            raise ValueError("exposure requires the poisson/log family")
        exposure = np.asarray(exposure, dtype=float)
        if exposure.shape != (rows,) or not np.isfinite(exposure).all() or np.any(exposure <= 0):
            raise ValueError("exposure must be finite, positive and match rows")
        result += np.log(exposure)
        if not np.isfinite(result).all():
            raise ValueError("combined offset and log exposure must be finite")
    return result


class GeneralizedLinearModel:
    """Weighted canonical mean NLL + l2/2 ||w||²; unpenalized intercept.

    family/link pairs: gaussian/identity, binomial/logit, poisson/log.
    link='auto' selects the canonical link. A matrix-free Newton-CG solver
    handles dense/CSR input. predict returns conditional means for every family.
    Offsets/exposures must be supplied again when predicting or scoring.
    """
    def __init__(self, *, family="gaussian", link="auto", l2=0., fit_intercept=True,
                 max_iter=100, max_cg_iter=200, tolerance=1e-7):
        self.options = _core.GLMOptions()
        for name in ("family", "link", "l2", "fit_intercept", "max_iter", "max_cg_iter", "tolerance"):
            setattr(self.options, name, locals()[name])

    def fit(self, X, y, *, sample_weight=None, offset=None, exposure=None):
        y = _targets(y)
        data = _dataset(X, y, binary_labels=False)
        options = _core.GLMOptions(self.options)
        offsets = _offset(offset, exposure, len(y), options.family)
        weights = [] if sample_weight is None else _weights(sample_weight, len(y))
        result = _core.fit_glm(data, options, weights, offsets)
        # Failed refits leave the previously fitted state intact.
        self._result = result
        self.n_features_in_ = len(result.weights)
        self.n_samples_ = len(y)
        self.coef_ = np.asarray(result.weights)
        self.intercept_ = result.intercept
        self.family_, self.link_ = result.family, result.link
        self.status_, self.converged_ = result.status, result.converged
        self.work_ = _work(result.work)
        self.history_ = [dict(iteration=h.iteration, loss=h.loss, gradient_norm=h.gradient_norm,
                              seconds=h.seconds, **_work(h.work)) for h in result.history]
        return self

    def _check_fitted(self):
        if not hasattr(self, "_result"):
            raise RuntimeError("fit must be called before prediction or deviance")

    def _predict(self, X, offset, exposure, linear):
        self._check_fitted()
        rows = X.shape[0] if hasattr(X, "shape") else len(X)
        offsets = _offset(offset, exposure, rows, self._result.family)
        return np.asarray(_core.predict_glm(_dataset(X), self._result, offsets, linear))

    def predict(self, X, *, offset=None, exposure=None):
        """Conditional mean: identity, event probability, or expected count."""
        return self._predict(X, offset, exposure, False)

    def decision_function(self, X, *, offset=None, exposure=None):
        """Linear predictor Xw + intercept + offset + log(exposure)."""
        return self._predict(X, offset, exposure, True)

    def predict_proba(self, X, *, offset=None, exposure=None):
        self._check_fitted()
        if self._result.family != "binomial":
            raise ValueError("predict_proba requires the binomial family")
        p = self.predict(X, offset=offset, exposure=exposure)
        return np.column_stack((1-p, p))

    def deviance(self, X, y, *, sample_weight=None, offset=None, exposure=None):
        """Weighted mean unit deviance (no regularization penalty)."""
        self._check_fitted()
        y = _targets(y)
        weights = [] if sample_weight is None else _weights(sample_weight, len(y))
        offsets = _offset(offset, exposure, len(y), self._result.family)
        return _core.glm_deviance(_dataset(X, y, binary_labels=False), self._result, weights, offsets)
