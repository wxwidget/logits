"""Finite targets and relative sample weights, independent of model classes."""

import numpy as np


def _targets(y):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not y.size or not np.isfinite(y).all():
        raise ValueError("targets must be a nonempty finite one-dimensional array")
    return y


def _weights(weights, size):
    if weights is None:
        return np.ones(size)
    weights = np.asarray(weights, dtype=float)
    if (
        weights.shape != (size,)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.any(weights > 0)
    ):
        raise ValueError(
            "sample weights must match targets, be finite/nonnegative and have positive mass"
        )
    # Scale invariance without overflow when raw weights are very large.
    return weights / weights.max()
