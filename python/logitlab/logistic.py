"""Python-facing binary classifier; optimization is owned by C++."""

import numpy as np
from . import _core
from ._data import _dataset, _work


class LogisticRegression:
    """Minimize mean binary log loss plus l2/2 times squared weight norm.

    Training runs in C++ with the GIL released. Input is copied into owned
    float64 dense/CSR storage; sparse matrices are never densified.
    """

    def __init__(
        self,
        *,
        solver="lbfgs",
        max_epochs=100,
        learning_rate=0.05,
        batch_size=256,
        l2=1e-4,
        tolerance=1e-6,
        history_size=10,
        seed=42,
        fit_intercept=True,
    ):
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
        self.history_ = [
            dict(
                epoch=h.epoch,
                loss=h.loss,
                gradient_norm=h.gradient_norm,
                seconds=h.seconds,
                **_work(h.work),
            )
            for h in result.history
        ]
        return self

    def predict_proba(self, X):
        if not hasattr(self, "_result"):
            raise RuntimeError("fit must be called before prediction")
        p = np.asarray(_core.predict(_dataset(X), self._result))
        return np.column_stack((1 - p, p))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)
