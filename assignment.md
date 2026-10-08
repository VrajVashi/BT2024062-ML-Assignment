# Assignment: Polynomial Regression — Agent Instructions

> This file is the single source of truth for completing the project. Read it fully before touching any code. Work autonomously, but follow the constraints exactly.

## 0. Context

- **Roll number:** `BT2024062`
- **Task:** Build polynomial regression models to predict a continuous target `y` for **two separate problems** (`var1`, `var2`) and produce predictions for the hidden test sets.
- **Constraint:** Use **only polynomial regression** (polynomial feature expansion + linear regression). No random forests, neural nets, gradient boosting, kernels/SVR, etc.
- **Metrics (hidden ground truth on test):** MSE (lower is better) and R² (higher is better).

### Files already in the repo root (do not rename or modify them)

```
BT2024062_train_var1.csv
BT2024062_test_var1.csv
BT2024062_train_var2.csv
BT2024062_test_var2.csv
sample_submission.csv
```

### Problem definitions

| Problem | Features | Max degree hint | Story |
|---|---|---|---|
| `var1` | `x1`–`x6` (6 features) | up to **10** | Steam turbine "Net Power Score" from six operational deviation parameters |
| `var2` | `x1`, `x2`, `x3` (3 features) | up to **20** | 3D spatial grid -> "Thermal Anomaly Score" |

- **Degree definition (important):** a polynomial of degree `d` means every term has *total degree* (sum of the powers of all features) **≤ d**. This is the same as `sklearn.preprocessing.PolynomialFeatures(degree=d)`. Do NOT use per-feature degree limits.
- The task statement says the true function is "of moderate degree (up to 10)" for var1 and "high degree (up to 20)" for var2, so the optimal degree is somewhere in `1..10` and `1..20` respectively. Do not assume the maximum is best.
- Train and test files are separate datasets. **Never** fit anything (scalers, degree selection, etc.) on test data. The test files have no `y`.

---

## 1. Deliverables (all three are required)

1. **Report (PDF)** — maximum 4–5 pages: approach, polynomial degree chosen for each problem, rationale, and any other techniques used.
2. **Prediction files** — exactly two CSVs in the repo root:
   - `BT2024062_pred_var1.csv`
   - `BT2024062_pred_var2.csv`
   
   Format must match `sample_submission.csv` **exactly** (same column names, same column order, same row order/IDs as the corresponding test file, same number of rows).
3. **GitHub repo** — all code used for training and inference, reproducible end-to-end.

---

## 2. Step-by-step plan

### Step 1 — Inspect everything first (no modelling yet)

- Print `head()`, `shape`, `dtypes`, `describe()`, and null counts for all 4 data CSVs.
- **Read `sample_submission.csv` carefully**: identify the exact column names, whether there is an ID/index column, and how it maps to the test files. Write what you find into `README.md`.
- Confirm column names of train vs test match (test has no `y`).
- Check for duplicates, NaNs, constant columns, and obvious outliers. The datasets are described as "preprocessed and cleaned," so do not aggressively drop data; if you do remove anything, justify it with numbers in the report.
- Check feature ranges/scales for each `x_i`.

### Step 2 — Set up the repo

Use this structure:

```
.
├── assignment.md
├── README.md
├── requirements.txt
├── data/                      # (optional) leave original CSVs in root if already there; do not duplicate large files unnecessarily
├── src/
│   ├── data.py                # loading + validation helpers
│   ├── model.py               # pipeline construction, degree selection
│   ├── select_degree.py       # CV sweep -> saves results + plots
│   ├── train_predict.py       # final fit + writes prediction CSVs
│   └── utils.py
├── results/
│   ├── cv_results_var1.csv
│   ├── cv_results_var2.csv
│   ├── degree_curve_var1.png
│   └── degree_curve_var2.png
├── report/
│   ├── report.md (or report.tex)
│   └── BT2024062_report.pdf
├── BT2024062_pred_var1.csv
└── BT2024062_pred_var2.csv
```

- Pin versions in `requirements.txt` (numpy, pandas, scikit-learn, scipy, matplotlib).
- Set a global random seed (`42`) everywhere. All results must be reproducible by running a documented command sequence.
- Add a `.gitignore` (Python caches, venvs, `.ipynb_checkpoints`). Do **not** ignore the datasets or the final prediction CSVs.

### Step 3 — Degree selection (the core of the assignment)

For **each problem independently**:

1. Hold out nothing from the test set; use only the train file.
2. Run **K-fold cross-validation** (`KFold(n_splits=5, shuffle=True, random_state=42)`; repeat with 10 folds or repeated K-fold as a sanity check) over degrees:
   - var1: `d = 1, 2, …, 10` (optionally 11–12 to show overfitting beyond the stated bound)
   - var2: `d = 1, 2, …, 20` (optionally up to 22–24)
3. For every degree record: **mean train MSE, mean validation MSE, validation std, mean validation R², number of polynomial terms.**
4. Save to `results/cv_results_var{1,2}.csv` and plot train-vs-validation MSE (log y-axis if needed) vs degree. Mark the chosen degree.
5. Choose the degree with the lowest mean CV MSE; if several are within ~1 standard error, prefer the **smaller** degree (parsimony). Document this rule.
6. Also report a single fixed 80/20 hold-out check at the chosen degree as an independent confirmation.

### Step 4 — Numerical stability (this will matter, especially for var2 degree ~15–20)

High-degree polynomial features are extremely ill-conditioned. Handle this properly:

- **Scale features before expanding**: e.g. `MinMaxScaler(feature_range=(-1, 1))` or `StandardScaler`, fit on train only, inside the sklearn `Pipeline` so CV does not leak.
- Prefer a numerically stable solver: `LinearRegression` (uses SVD/lstsq) is fine; avoid explicit normal-equation inversion (`inv(X.T @ X)`).
- Optional, if raw monomials still blow up at high degree: use an **orthogonal polynomial basis** (Legendre/Chebyshev tensor-product restricted to total degree ≤ d). This is still polynomial regression of the same degree and is allowed — document it clearly if used.
- Check the condition number / rank of the design matrix and warn if rank-deficient. Number of terms for reference: var1 at d=10 is `C(16,6)=8008`; var2 at d=20 is `C(23,3)=1771`. Make sure `n_samples` comfortably exceeds the number of terms for the chosen degree; if it does not, say so in the report.
- Do NOT silently pick a degree where the fit has numerical blow-ups (huge coefficients, NaNs, negative R² on validation).

### Step 5 — Optional regularisation (only if justified)

- Plain OLS at the CV-selected degree is the **primary** model.
- If OLS clearly overfits at the best degree (large CV gap, unstable coefficients), you may add **Ridge (L2)** on top of the polynomial features and tune `alpha` by CV. This is still polynomial regression, but you **must** state it explicitly in the report as an additional technique.
- Compare OLS vs Ridge in a small table (CV MSE and R²). Use whichever generalises better; keep the simpler one on ties.
- Do not use Lasso/ElasticNet for var1 d=10 unless runtime is acceptable; if used, document it.

### Step 6 — Final fit and inference

- Refit the chosen pipeline on the **entire** training file for each problem.
- Predict on the corresponding test file.
- Write `BT2024062_pred_var1.csv` and `BT2024062_pred_var2.csv` in the **exact format of `sample_submission.csv`**.
- Run these assertions in code and fail loudly if they break:
  - row count equals test file row count
  - no NaN / inf in predictions
  - column names/order equal the sample submission
  - IDs (if any) match the test file in the same order
  - the files are named exactly `BT2024062_pred_var1.csv` and `BT2024062_pred_var2.csv`
- Print summary stats of predictions vs train `y` (min/max/mean/std) as a sanity check — predictions wildly outside the train `y` range indicate extrapolation blow-up and must be investigated.

### Step 7 — Report (PDF, ≤ 5 pages)

Write `report/report.md` (or LaTeX) and export to `report/BT2024062_report.pdf`. Include:

1. **Problem & data overview** — sizes, features, target, anything noteworthy from EDA.
2. **Method** — pipeline: scaling -> polynomial features (total-degree definition) -> linear regression (and Ridge if used); CV protocol; degree-selection rule.
3. **Results per problem** — table of CV MSE/R² across degrees, the degree-vs-error plot, **the chosen degree**, and the rationale (underfitting for low degrees, overfitting for high degrees, 1-SE rule).
4. **Numerical considerations** — scaling, conditioning, number of terms vs samples.
5. **Final metrics** — CV MSE and R² (and hold-out MSE/R²) for the final models. State clearly that test metrics are unknown because the ground truth is hidden.
6. **Conclusion / limitations.**

Keep it concise: plots should be small, tables compact. Verify the PDF page count is ≤ 5.

### Step 8 — README and final repo check

`README.md` must contain: project description, environment setup, exact commands to reproduce (e.g. `python src/select_degree.py` then `python src/train_predict.py`), chosen degrees, and a file map.

Final checklist before declaring done:

- [ ] Both prediction CSVs exist, correctly named, match the sample format
- [ ] Code runs from a clean environment using `requirements.txt`
- [ ] Degree-selection results and plots saved in `results/`
- [ ] Report PDF exists and is ≤ 5 pages
- [ ] No test-set information was used in fitting or selection
- [ ] Only polynomial regression (± Ridge, documented) is used
- [ ] Repo is committed (do not push to a remote unless the user asks; tell the user the commands to do so)

---

## 3. Rules and pitfalls

- **No data leakage:** scalers and feature expansion live inside a `Pipeline` that is fit within each CV fold.
- **Never tune on test data.** There are no test labels anyway; do not try to infer them.
- **Do not fabricate numbers.** Every number in the report must come from code you actually ran. If something fails or can't be run, say so.
- **Do not change the original CSV files.**
- Keep the code clean, commented, and runnable from the repo root.
- If anything in `sample_submission.csv` contradicts this file (columns, IDs, ordering), **the sample submission wins** — note the discrepancy in the README.
- If CV is too slow at high degrees, reduce cost with fewer folds or by using the SVD/`lstsq` solver with float64, but keep results reproducible. Do not skip degrees silently.

## 4. When finished, report back with

1. The chosen degree (and Ridge alpha if used) for var1 and var2
2. CV MSE and R² for each final model
3. Paths to the two prediction files and the report PDF
4. Any assumptions made or issues encountered
