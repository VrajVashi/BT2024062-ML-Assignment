import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge


def build_ols_pipeline(degree: int) -> Pipeline:
    return Pipeline([
        ("scaler", MinMaxScaler(feature_range=(-1, 1))),
        ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
        ("lr", LinearRegression(fit_intercept=False)),
    ])


def build_ridge_pipeline(degree: int, alpha: float) -> Pipeline:
    return Pipeline([
        ("scaler", MinMaxScaler(feature_range=(-1, 1))),
        ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
        ("ridge", Ridge(alpha=alpha, fit_intercept=False)),
    ])


def n_poly_terms(n_features: int, degree: int) -> int:
    from math import comb
    return comb(n_features + degree, degree)


def check_condition_number(X_poly: np.ndarray) -> float:
    try:
        sv = np.linalg.svd(X_poly, compute_uv=False)
        sv_nonzero = sv[sv > 0]
        cond = sv_nonzero[0] / sv_nonzero[-1]
        return float(cond)
    except Exception:
        return float("inf")
