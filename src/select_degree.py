import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import mean_squared_error, r2_score

from src.data import load_train
from src.model import build_ols_pipeline, build_ridge_pipeline, n_poly_terms
from src.utils import set_global_seed

set_global_seed()

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

VAR1_DEGREES = list(range(1, 13))
VAR2_DEGREES = list(range(1, 25))

KF5 = KFold(n_splits=5, shuffle=True, random_state=42)
KF10 = KFold(n_splits=10, shuffle=True, random_state=42)

VAR1_RIDGE_ALPHA = 1.0
VAR2_RIDGE_ALPHA = 1.0


def neg_mse_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return -mean_squared_error(y, y_pred)


def r2_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return r2_score(y, y_pred)


def run_cv_sweep(X: np.ndarray, y: np.ndarray, degrees: list, variant: str, alpha: float = None):
    print(f"\n{'='*60}")
    print(f"  CV Sweep: {variant}  (n={len(y)}, feats={X.shape[1]})")
    print(f"{'='*60}")

    records = []
    for d in degrees:
        terms = n_poly_terms(X.shape[1], d)
        if alpha is not None:
            pipe = build_ridge_pipeline(d, alpha)
        else:
            pipe = build_ols_pipeline(d)

        cv5 = cross_validate(
            pipe, X, y,
            cv=KF5,
            scoring={"neg_mse": neg_mse_scorer, "r2": r2_scorer},
            return_train_score=True,
        )

        val_mse_mean = -cv5["test_neg_mse"].mean()
        val_mse_std = cv5["test_neg_mse"].std()
        train_mse_mean = -cv5["train_neg_mse"].mean()
        val_r2_mean = cv5["test_r2"].mean()

        records.append({
            "degree": d,
            "n_terms": terms,
            "mean_train_mse": train_mse_mean,
            "mean_val_mse": val_mse_mean,
            "std_val_mse": val_mse_std,
            "mean_val_r2": val_r2_mean,
        })

        print(
            f"  d={d:2d}  terms={terms:5d}  "
            f"train_mse={train_mse_mean:.4f}  "
            f"val_mse={val_mse_mean:.4f}±{val_mse_std:.4f}  "
            f"val_r2={val_r2_mean:.4f}"
        )

    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / f"cv_results_{variant}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CV results: {csv_path}")
    return df


def select_degree(df: pd.DataFrame) -> int:
    best_idx = df["mean_val_mse"].idxmin()
    best_mse = df.loc[best_idx, "mean_val_mse"]
    best_std = df.loc[best_idx, "std_val_mse"]
    n_folds = 5
    se = best_std / np.sqrt(n_folds)
    threshold = best_mse + se
    candidates = df[df["mean_val_mse"] <= threshold]
    chosen_idx = candidates["degree"].idxmin()
    chosen_degree = int(candidates.loc[chosen_idx, "degree"])
    print(f"\nDegree selection:")
    print(f"  Best MSE = {best_mse:.4f} at degree {int(df.loc[best_idx, 'degree'])}")
    print(f"  1-SE threshold = {threshold:.4f}")
    print(f"  Chosen degree (parsimonious) = {chosen_degree}")
    return chosen_degree


def plot_degree_curve(df: pd.DataFrame, variant: str, chosen_degree: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["degree"], df["mean_train_mse"], marker="o", label="Train MSE", color="#2196F3")
    ax.plot(df["degree"], df["mean_val_mse"], marker="s", label="Val MSE (5-fold)", color="#F44336")
    ax.fill_between(
        df["degree"],
        df["mean_val_mse"] - df["std_val_mse"],
        df["mean_val_mse"] + df["std_val_mse"],
        alpha=0.2, color="#F44336"
    )
    ax.axvline(x=chosen_degree, linestyle="--", color="green", linewidth=1.5,
               label=f"Chosen d={chosen_degree}")
    ax.set_yscale("log")
    ax.set_xlabel("Polynomial Degree", fontsize=12)
    ax.set_ylabel("MSE (log scale)", fontsize=12)
    ax.set_title(f"{variant} — Degree vs MSE", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / f"degree_curve_{variant}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out}")


def compare_ols_ridge(X, y, degree, alphas, variant, cv):
    print(f"\n  OLS vs Ridge comparison for {variant} d={degree}")
    results = []

    ols_pipe = build_ols_pipeline(degree)
    cv_ols = cross_validate(
        ols_pipe, X, y, cv=cv,
        scoring={"neg_mse": neg_mse_scorer, "r2": r2_scorer},
        return_train_score=True,
    )
    results.append({
        "model": "OLS",
        "alpha": None,
        "cv_mse": -cv_ols["test_neg_mse"].mean(),
        "cv_r2": cv_ols["test_r2"].mean(),
    })
    print(f"    OLS  cv_mse={results[-1]['cv_mse']:.4f}  cv_r2={results[-1]['cv_r2']:.4f}")

    for a in alphas:
        ridge_pipe = build_ridge_pipeline(degree, a)
        cv_r = cross_validate(
            ridge_pipe, X, y, cv=cv,
            scoring={"neg_mse": neg_mse_scorer, "r2": r2_scorer},
            return_train_score=True,
        )
        results.append({
            "model": "Ridge",
            "alpha": a,
            "cv_mse": -cv_r["test_neg_mse"].mean(),
            "cv_r2": cv_r["test_r2"].mean(),
        })
        print(f"    Ridge(alpha={a:.1e}) cv_mse={results[-1]['cv_mse']:.4f}  cv_r2={results[-1]['cv_r2']:.4f}")

    return pd.DataFrame(results)


def holdout_check(X, y, degree, pipeline, variant):
    from sklearn.model_selection import train_test_split
    X_tr, X_ho, y_tr, y_ho = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_ho)
    mse = mean_squared_error(y_ho, y_pred)
    r2 = r2_score(y_ho, y_pred)
    print(f"  Holdout (80/20) for {variant} d={degree}: MSE={mse:.4f}  R²={r2:.4f}")
    return mse, r2


def main():
    print("\n" + "="*60)
    print("  DEGREE SELECTION — VAR1")
    print("="*60)
    X1, y1 = load_train("var1")

    df1_ols = run_cv_sweep(X1.values, y1.values, VAR1_DEGREES, "var1_ols", alpha=None)
    chosen_d1_ols = select_degree(df1_ols)

    alphas1 = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    comp1 = compare_ols_ridge(X1.values, y1.values, chosen_d1_ols, alphas1, "var1", KF5)
    print(comp1.to_string(index=False))

    best1 = comp1.loc[comp1["cv_mse"].idxmin()]
    if best1["model"] == "Ridge":
        VAR1_RIDGE_ALPHA_FINAL = best1["alpha"]
        df1_final = run_cv_sweep(X1.values, y1.values, VAR1_DEGREES, "var1", alpha=VAR1_RIDGE_ALPHA_FINAL)
        chosen_d1 = select_degree(df1_final)
        final_pipe1 = build_ridge_pipeline(chosen_d1, VAR1_RIDGE_ALPHA_FINAL)
        model1_type = f"Ridge(alpha={VAR1_RIDGE_ALPHA_FINAL})"
    else:
        df1_final = df1_ols
        chosen_d1 = chosen_d1_ols
        final_pipe1 = build_ols_pipeline(chosen_d1)
        model1_type = "OLS"

    plot_degree_curve(df1_final, "var1", chosen_d1)
    holdout_check(X1.values, y1.values, chosen_d1, final_pipe1, "var1")

    print(f"\n  VAR1 FINAL: degree={chosen_d1}, model={model1_type}")

    print("\n" + "="*60)
    print("  DEGREE SELECTION — VAR2")
    print("="*60)
    X2, y2 = load_train("var2")

    df2_ols = run_cv_sweep(X2.values, y2.values, VAR2_DEGREES, "var2_ols", alpha=None)
    chosen_d2_ols = select_degree(df2_ols)

    alphas2 = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    comp2 = compare_ols_ridge(X2.values, y2.values, chosen_d2_ols, alphas2, "var2", KF5)
    print(comp2.to_string(index=False))

    best2 = comp2.loc[comp2["cv_mse"].idxmin()]
    if best2["model"] == "Ridge":
        VAR2_RIDGE_ALPHA_FINAL = best2["alpha"]
        df2_final = run_cv_sweep(X2.values, y2.values, VAR2_DEGREES, "var2", alpha=VAR2_RIDGE_ALPHA_FINAL)
        chosen_d2 = select_degree(df2_final)
        final_pipe2 = build_ridge_pipeline(chosen_d2, VAR2_RIDGE_ALPHA_FINAL)
        model2_type = f"Ridge(alpha={VAR2_RIDGE_ALPHA_FINAL})"
    else:
        df2_final = df2_ols
        chosen_d2 = chosen_d2_ols
        final_pipe2 = build_ols_pipeline(chosen_d2)
        model2_type = "OLS"

    plot_degree_curve(df2_final, "var2", chosen_d2)
    holdout_check(X2.values, y2.values, chosen_d2, final_pipe2, "var2")

    print(f"\n  VAR2 FINAL: degree={chosen_d2}, model={model2_type}")

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"  var1: degree={chosen_d1}, model={model1_type}")
    print(f"  var2: degree={chosen_d2}, model={model2_type}")

    results_summary = {
        "var1_degree": chosen_d1,
        "var1_model": model1_type,
        "var2_degree": chosen_d2,
        "var2_model": model2_type,
    }

    summary_path = RESULTS_DIR / "selection_summary.csv"
    pd.DataFrame([results_summary]).to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
