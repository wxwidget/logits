import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import refactor
from benchmarks import refactor_report


def test_default_bounds_preserve_unconstrained_and_intercept_contract():
    from logitlab import LinearRegression

    X = np.arange(12.0).reshape(6, 2)
    y = X @ [0.5, -0.25] + 2.0
    implicit = LinearRegression(l2=0.1).fit(X, y)
    explicit = LinearRegression(l2=0.1, bounds=(-np.inf, np.inf)).fit(X, y)
    np.testing.assert_allclose(implicit.coef_, explicit.coef_)
    assert implicit._unconstrained and explicit._unconstrained
    constrained = LinearRegression(l2=0.1, intercept_bounds=(0.0, 0.5)).fit(X, y)
    assert not constrained._unconstrained
    assert 0.0 <= constrained.intercept_ <= 0.5


def paired_rows():
    return [
        dict(
            case=case,
            storage=storage,
            seed=42,
            repeat=0,
            secondary=case in refactor.SECONDARY,
            converged=case not in refactor.SECONDARY,
            stationarity=1e-7,
            objective=0.5,
            heldout_quality=0.2,
            data_sha256="same",
            fit_seconds=2.0,
        )
        for case in refactor.PRIMARY + refactor.SECONDARY
        for storage in ("dense", "csr")
    ]


def test_performance_gate_and_accuracy_guard():
    before = paired_rows()
    after = [row | dict(fit_seconds=1.0) for row in before]
    result = refactor.compare(before, after)
    assert result["passed"] and result["primary_geometric_speedup"] == pytest.approx(
        2.0
    )
    assert len(result["cases"]) == 22
    for key, value in (
        ("objective", 0.6),
        ("converged", False),
        ("data_sha256", "different"),
    ):
        changed = [dict(r) for r in after]
        changed[0][key] = value
        with pytest.raises(ValueError):
            refactor.compare(before, changed)
    with pytest.raises(ValueError):
        refactor.compare(before, after[:-1])


def test_report_rejects_quick_underpowered_and_incomplete_evidence():
    before = [
        row | dict(seed=seed, repeat=repeat)
        for row in paired_rows()
        for seed in (42, 43, 44)
        for repeat in range(3)
    ]
    after = [row | dict(fit_seconds=1.0) for row in before]
    data = dict(
        eligible_for_performance_claim=True,
        config=dict(quick=False, seeds=[42, 43, 44], repeats=3),
        before=before,
        after=after,
    )
    assert refactor_report.validate(data)["passed"]
    for config in (
        dict(quick=True, seeds=[42, 43, 44], repeats=3),
        dict(quick=False, seeds=[42], repeats=3),
        dict(quick=False, seeds=[42, 43, 44], repeats=1),
    ):
        with pytest.raises(ValueError):
            refactor_report.validate(data | dict(config=config))
    with pytest.raises(ValueError):
        refactor_report.validate(data | dict(after=after[:-1]))


def test_independent_smooth_and_nonsmooth_stationarity():
    X = np.eye(3)
    y = np.array([1.0, 2.0, 3.0])
    theta = np.r_[y, 0.0]
    loss, norm = refactor.independent(X, y, theta, "gaussian", 0.0, 0.0)
    assert loss == 0 and norm == 0


def test_models_import_without_circular_dependencies():
    for name in (
        "logitlab.logistic",
        "logitlab.linear",
        "logitlab.glm",
        "logitlab._data",
    ):
        assert importlib.import_module(name)


def test_validated_binding_still_checks_missing_and_wrong_targets():
    from logitlab import _core

    no_targets = _core.Dataset.dense([[0.0], [1.0]])
    for fit, options in (
        (_core.fit, _core.Options()),
        (_core.fit_linear, _core.LinearOptions()),
        (_core.fit_glm, _core.GLMOptions()),
    ):
        with pytest.raises(ValueError, match="label count"):
            fit(no_targets, options)
    nonbinary = _core.Dataset.dense([[0.0], [1.0]], [2.0, 3.0], False)
    with pytest.raises(ValueError, match="labels"):
        _core.fit(nonbinary, _core.Options())


@pytest.mark.parametrize("family", ["gaussian", "binomial", "poisson"])
def test_wide_sparse_kernel_with_zero_mass_extreme_row(family):
    from scipy import sparse
    from scipy.special import expit
    from logitlab import GeneralizedLinearModel

    rng = np.random.default_rng(92)
    X = sparse.random(
        1200,
        128,
        density=0.5,
        format="csr",
        random_state=rng,
        data_rvs=lambda n: rng.normal(size=n),
    )
    offset = rng.normal(scale=0.1, size=X.shape[0])
    eta = np.asarray(X @ rng.normal(size=X.shape[1])).ravel() / 20 + offset
    y = (
        eta + rng.normal(size=len(eta))
        if family == "gaussian"
        else rng.binomial(1, expit(eta)).astype(float)
        if family == "binomial"
        else rng.poisson(np.exp(eta)).astype(float)
    )
    weights = np.linspace(0, 2, len(y))
    weights[::17] = 0
    X.data[X.indptr[0]] = 1e308
    offset[0] = 1000
    model = GeneralizedLinearModel(family=family, l2=0.1, tolerance=1e-7).fit(
        X, y, sample_weight=weights, offset=offset
    )
    mask = weights > 0
    q = weights[mask] / weights[mask].sum()
    z = np.asarray(X[mask] @ model.coef_).ravel() + model.intercept_ + offset[mask]
    mu = z if family == "gaussian" else expit(z) if family == "binomial" else np.exp(z)
    residual = q * (mu - y[mask])
    gradient = np.r_[
        np.asarray(X[mask].T @ residual).ravel() + 0.1 * model.coef_, residual.sum()
    ]
    assert model.converged_ and np.linalg.norm(gradient) <= 1.0001e-7


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_csr_binding_owns_and_validates_integer_storage(dtype, strided):
    from logitlab import _core

    indices = np.array([0, 1], dtype=dtype)
    offsets = np.array([0, 1, 2], dtype=dtype)
    if strided:
        indices = np.repeat(indices, 2)[::2]
        offsets = np.repeat(offsets, 2)[::2]
    data = _core.Dataset.csr([1.0, 1.0], indices, offsets, 2, 2, [0.0, 1.0])
    before = _core.predict(data, _core.fit(data, _core.Options()))
    indices[:] = 99
    offsets[:] = 99  # Later caller mutation cannot change owned data.
    np.testing.assert_allclose(
        _core.predict(data, _core.fit(data, _core.Options())), before
    )
    with pytest.raises(ValueError):
        _core.Dataset.csr([1.0], np.array([-1], dtype=dtype), [0, 1], 1, 2, [0.0])
