"""Owned input conversion and work records shared by estimator interfaces."""

import numpy as np
from scipy import sparse
from . import _core


def _work(work):
    return {
        key: getattr(work, key)
        for key in (
            "full_evaluations",
            "batch_gradients",
            "hessian_vector_products",
            "row_visits",
            "parameter_updates",
        )
    }


def _dataset(X, y=None, binary_labels=True):
    if sparse.issparse(X):
        # Validate compressed storage before SciPy canonicalization can dereference indices.
        owned = X.copy()
        checked = hasattr(owned, "check_format")
        if checked:
            owned.check_format(full_check=True)
        X = owned.tocsr(copy=False)
        # CSR conversion can return the already checked object unchanged.
        # Other formats produce new compressed storage that must be checked.
        if X is not owned or not checked:
            X.check_format(full_check=True)
        X.sum_duplicates()
        X.sort_indices()
        return _core.Dataset.csr(
            X.data, X.indices, X.indptr, *X.shape, y, binary_labels
        )
    return _core.Dataset.dense(np.asarray(X, dtype=np.float64), y, binary_labels)
