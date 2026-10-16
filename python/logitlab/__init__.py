"""C++ linear and logistic regression with NumPy/SciPy interfaces."""
import numpy as np
from scipy import sparse
from . import _core

__all__ = ["LogisticRegression", "LinearRegression", "Ridge", "Lasso", "ElasticNet",
           "regression_metrics"]

def _work(work):
    return {key: getattr(work, key) for key in (
        "full_evaluations", "batch_gradients", "hessian_vector_products",
        "row_visits", "parameter_updates")}

def _dataset(X, y=None, binary_labels=True):
    if sparse.issparse(X):
        # Validate compressed storage before SciPy canonicalization can dereference indices.
        X = X.copy()
        if hasattr(X, "check_format"):
            X.check_format(full_check=True)
        X = X.tocsr(copy=False)
        X.check_format(full_check=True)
        X.sum_duplicates()
        X.sort_indices()
        return _core.Dataset.csr(X.data, X.indices, X.indptr, *X.shape, y, binary_labels)
    return _core.Dataset.dense(np.asarray(X, dtype=np.float64), y, binary_labels)

class LogisticRegression:
    """Minimize mean binary log loss plus l2/2 times squared weight norm.

    Training runs in C++ with the GIL released. Input is copied into owned
    float64 dense/CSR storage; sparse matrices are never densified.
    """
    def __init__(self, *, solver="lbfgs", max_epochs=100, learning_rate=0.05,
                 batch_size=256, l2=1e-4, tolerance=1e-6, history_size=10,
                 seed=42, fit_intercept=True):
        self.options = _core.Options()
        for key, value in locals().copy().items():
            if key != "self":
                setattr(self.options, key, value)

    def fit(self, X, y):
        result = _core.fit(_dataset(X, np.asarray(y, dtype=np.float64)), self.options)
        self._result = result
        self.coef_ = np.asarray(result.weights)
        self.intercept_ = result.intercept
        self.converged_ = result.converged
        self.status_ = result.status
        self.work_ = _work(result.work)
        self.history_ = [dict(epoch=h.epoch, loss=h.loss,
                              gradient_norm=h.gradient_norm, seconds=h.seconds,
                              **_work(h.work))
                         for h in result.history]
        return self

    def predict_proba(self, X):
        if not hasattr(self, "_result"):
            raise RuntimeError("fit must be called before prediction")
        p = np.asarray(_core.predict(_dataset(X), self._result))
        return np.column_stack((1 - p, p))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)


# Loaded after conversion helpers to avoid circular initialization.
from .analysis import regression_metrics
from .linear import ElasticNet, Lasso, LinearRegression, Ridge
