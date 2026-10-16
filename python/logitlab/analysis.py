"""Regression metrics and explicitly restricted OLS/WLS inference."""
import numpy as np
from scipy import sparse, stats


def _targets(y):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not y.size or not np.isfinite(y).all():
        raise ValueError("targets must be a nonempty finite one-dimensional array")
    return y


def _weights(weights, size):
    if weights is None:
        return np.ones(size)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (size,) or not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("sample weights must match targets, be finite/nonnegative and have positive mass")
    # Scale invariance without overflow when raw weights are very large.
    return weights / weights.max()


def regression_metrics(y_true, y_pred, *, sample_weight=None):
    """Weighted MSE, RMSE, MAE, centered R² and explained variance.

    Constant targets use finite R²: 1 for exact prediction, otherwise 0.
    Zero-weight rows do not affect metrics; at least one positive weight is required.
    """
    y, prediction = _targets(y_true), _targets(y_pred)
    if y.shape != prediction.shape:
        raise ValueError("predictions must match targets")
    weights = _weights(sample_weight, len(y))
    mask = weights > 0
    y, prediction, weights = y[mask], prediction[mask], weights[mask]
    q = weights / weights.sum()
    residual = y - prediction
    mse = float(q @ (residual**2))
    variance = 0. if np.all(y == y[0]) else float(q @ ((y - q @ y)**2))
    residual_variance = float(q @ ((residual - q @ residual)**2))
    result = dict(mse=mse, rmse=float(np.sqrt(mse)), mae=float(q @ np.abs(residual)),
                  r2=float(1 - mse / variance) if variance else float(mse == 0),
                  explained_variance=float(1 - residual_variance / variance) if variance
                  else float(residual_variance == 0))
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("regression metrics overflowed; scale targets and predictions")
    return result


def ols_summary(X, y, coef, intercept, *, fit_intercept, sample_weight,
                confidence, covariance, max_features, fit_tolerance):
    """SVD-based full-rank OLS/WLS inference, without n-by-n hat matrices."""
    if sparse.issparse(X):
        raise ValueError("OLS inference requires explicit dense input; use evaluate for sparse data")
    X, y = np.asarray(X, dtype=float), _targets(y)
    if (X.ndim != 2 or X.shape != (len(y), len(coef)) or not np.isfinite(X).all()):
        raise ValueError("invalid inference design or target shape")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if covariance not in ("classical", "HC3"):
        raise ValueError("covariance must be classical or HC3")
    if not isinstance(max_features, (int, np.integer)) or not 0 < X.shape[1] <= max_features:
        raise ValueError("inference exceeds max_features")
    if X.shape[0] * (X.shape[1] + int(fit_intercept)) > 5_000_000:
        raise ValueError("dense inference exceeds 5 million design elements")
    weights = _weights(sample_weight, len(y))
    mask = weights > 0
    X, y, weights = X[mask], y[mask], weights[mask]
    design = np.column_stack((X, np.ones(len(y)))) if fit_intercept else X
    theta = np.r_[coef, intercept] if fit_intercept else np.asarray(coef)
    weighted_design = design * np.sqrt(weights[:, None])
    u, singular, vt = np.linalg.svd(weighted_design, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * max(design.shape) * np.finfo(float).eps))
    nobs, parameters = design.shape
    df = nobs - rank
    if rank != parameters or df <= 0:
        raise ValueError("OLS inference requires full rank and positive residual degrees of freedom")
    fitted = design @ theta
    residual = y - fitted
    gradient = design.T @ (weights * residual) / weights.sum()
    if np.linalg.norm(gradient) > max(10 * fit_tolerance, 1e-7):
        raise ValueError("summary requires a least-squares fit on these data and weights")
    sse = float(weights @ residual**2)
    sigma2 = sse / df
    if not np.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError("positive residual variance required for inference")
    inverse = (vt.T / singular**2) @ vt
    leverage = np.sum(u**2, axis=1)
    if covariance == "classical":
        cov = sigma2 * inverse
        critical = stats.t.ppf((1 + confidence) / 2, df)
        survival = lambda value: stats.t.sf(value, df)
    else:
        if np.any(1 - leverage <= 1e-12):
            raise ValueError("HC3 is undefined for unit leverage")
        adjusted = np.sqrt(weights) * residual / (1 - leverage)
        middle = weighted_design.T @ (adjusted[:, None]**2 * weighted_design)
        cov = inverse @ middle @ inverse
        critical = stats.norm.ppf((1 + confidence) / 2)
        survival = stats.norm.sf
    standard_error = np.sqrt(np.maximum(np.diag(cov), 0))
    statistics = np.divide(theta, standard_error, out=np.full(parameters, np.nan), where=standard_error > 0)
    names = [f"x{j}" for j in range(X.shape[1])] + (["intercept"] if fit_intercept else [])
    coefficients = [dict(name=name, estimate=float(estimate), standard_error=float(error),
                         statistic=float(statistic), p_value=float(2 * survival(abs(statistic))),
                         ci_lower=float(estimate - critical * error), ci_upper=float(estimate + critical * error))
                    for name, estimate, error, statistic in zip(names, theta, standard_error, statistics)]
    center = float(weights @ y / weights.sum()) if fit_intercept else 0.
    total = float(weights @ (y - center)**2)
    r2 = 1 - sse / total if total else float(sse == 0)
    adjusted_r2 = 1 - (1 - r2) * (nobs - int(fit_intercept)) / df
    df_model = parameters - int(fit_intercept)
    f_statistic = max(0., (total - sse) / df_model) / sigma2 if covariance == "classical" else None
    f_pvalue = float(stats.f.sf(f_statistic, df_model, df)) if f_statistic is not None else None
    log_likelihood = -.5 * nobs * (np.log(2 * np.pi) + 1 + np.log(sse / nobs)) + .5 * np.log(weights).sum()
    denominator = np.maximum(1 - leverage, 0)
    cooks = np.divide(weights * residual**2 * leverage, parameters * sigma2 * denominator**2,
                      out=np.full(nobs, np.nan), where=denominator > 1e-12)
    return dict(nobs=nobs, rank=rank, df_model=df_model, df_resid=df, r2=float(r2),
                adjusted_r2=float(adjusted_r2), residual_variance=sigma2,
                condition_number=float(singular[0] / singular[-1]), covariance=cov,
                covariance_type=covariance, confidence=confidence, coefficients=coefficients,
                f_statistic=f_statistic, f_pvalue=f_pvalue, log_likelihood=float(log_likelihood),
                aic=float(-2 * log_likelihood + 2 * parameters),
                bic=float(-2 * log_likelihood + np.log(nobs) * parameters),
                fitted_values=fitted, residuals=residual, leverage=leverage, cooks_distance=cooks)
