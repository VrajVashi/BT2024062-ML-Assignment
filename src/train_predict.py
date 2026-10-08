import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, KFold

from src.data import load_train, load_test, validate_predictions, write_predictions
from src.model import build_ols_pipeline, build_ridge_pipeline
from src.utils import set_global_seed

set_global_seed()

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

VAR1_DEGREE = 5
VAR1_MODEL = "Ridge"
VAR1_ALPHA = 10.0

VAR2_DEGREE = 10
VAR2_MODEL = "Ridge"
VAR2_ALPHA = 1.0


def neg_mse_scorer(estimator, X, y):
    return -mean_squared_error(y, estimator.predict(X))


def r2_scorer(estimator, X, y):
    return r2_score(y, estimator.predict(X))


def build_final_pipeline(degree, model_type, alpha):
    if model_type == "Ridge":
        return build_ridge_pipeline(degree, alpha)
    return build_ols_pipeline(degree)


def run_variant(variant: str, degree: int, model_type: str, alpha: float):
    print(f"\n{'='*60}")
    print(f"  FINAL FIT — {variant.upper()}  degree={degree}  model={model_type}" +
          (f"(alpha={alpha})" if model_type == "Ridge" else ""))
    print(f"{'='*60}")

    X_train, y_train = load_train(variant)
    X_test = load_test(variant)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pipe_cv = build_final_pipeline(degree, model_type, alpha)
    cv_res = cross_validate(
        pipe_cv, X_train.values, y_train.values, cv=kf,
        scoring={"neg_mse": neg_mse_scorer, "r2": r2_scorer},
        return_train_score=True,
    )
    cv_mse = -cv_res["test_neg_mse"].mean()
    cv_r2 = cv_res["test_r2"].mean()
    print(f"  CV (5-fold) MSE = {cv_mse:.6f}   R² = {cv_r2:.6f}")

    pipe_final = build_final_pipeline(degree, model_type, alpha)
    pipe_final.fit(X_train.values, y_train.values)

    y_train_pred = pipe_final.predict(X_train.values)
    train_mse = mean_squared_error(y_train.values, y_train_pred)
    train_r2 = r2_score(y_train.values, y_train_pred)
    print(f"  Train MSE = {train_mse:.6f}   Train R² = {train_r2:.6f}")

    y_pred = pipe_final.predict(X_test.values)

    print(f"\n  Prediction sanity check vs train y:")
    print(f"    Train y  — min={y_train.min():.4f}  max={y_train.max():.4f}  "
          f"mean={y_train.mean():.4f}  std={y_train.std():.4f}")
    print(f"    Pred y   — min={y_pred.min():.4f}  max={y_pred.max():.4f}  "
          f"mean={y_pred.mean():.4f}  std={y_pred.std():.4f}")

    extrapolation_ratio = (np.sum(y_pred < y_train.min()) + np.sum(y_pred > y_train.max())) / len(y_pred)
    if extrapolation_ratio > 0.1:
        print(f"  WARNING: {extrapolation_ratio:.1%} predictions outside train y range — check for blow-up!")
    else:
        print(f"  OK: only {extrapolation_ratio:.1%} predictions outside train y range")

    validate_predictions(y_pred, X_test, variant)
    write_predictions(y_pred, variant)

    return cv_mse, cv_r2


def main():
    print("\nLoading selection summary if available...")
    summary_path = RESULTS_DIR / "selection_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path).iloc[0]
        v1_degree = int(summary["var1_degree"])
        v1_model_str = str(summary["var1_model"])
        v2_degree = int(summary["var2_degree"])
        v2_model_str = str(summary["var2_model"])

        if "Ridge" in v1_model_str:
            v1_model = "Ridge"
            try:
                v1_alpha = float(v1_model_str.split("alpha=")[1].rstrip(")"))
            except Exception:
                v1_alpha = VAR1_ALPHA
        else:
            v1_model = "OLS"
            v1_alpha = None

        if "Ridge" in v2_model_str:
            v2_model = "Ridge"
            try:
                v2_alpha = float(v2_model_str.split("alpha=")[1].rstrip(")"))
            except Exception:
                v2_alpha = VAR2_ALPHA
        else:
            v2_model = "OLS"
            v2_alpha = None
    else:
        print("No selection_summary.csv found — using hardcoded defaults.")
        v1_degree, v1_model, v1_alpha = VAR1_DEGREE, VAR1_MODEL, VAR1_ALPHA
        v2_degree, v2_model, v2_alpha = VAR2_DEGREE, VAR2_MODEL, VAR2_ALPHA

    mse1, r2_1 = run_variant("var1", v1_degree, v1_model, v1_alpha)
    mse2, r2_2 = run_variant("var2", v2_degree, v2_model, v2_alpha)

    print("\n" + "="*60)
    print("  FINAL SUMMARY")
    print("="*60)
    print(f"  var1:  degree={v1_degree}  model={v1_model}  CV_MSE={mse1:.4f}  CV_R²={r2_1:.4f}")
    print(f"  var2:  degree={v2_degree}  model={v2_model}  CV_MSE={mse2:.4f}  CV_R²={r2_2:.4f}")
    print(f"\n  Prediction files:")
    print(f"    {ROOT / 'BT2024062_pred_var1.csv'}")
    print(f"    {ROOT / 'BT2024062_pred_var2.csv'}")


if __name__ == "__main__":
    main()
