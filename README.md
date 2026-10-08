# Polynomial Regression Assignment — BT2024062

## Project Description

Two polynomial regression models that predict a continuous target `y` for:

- **var1** — "Net Power Score" for a steam turbine, from 6 operational deviation parameters (`x1`–`x6`)
- **var2** — "Thermal Anomaly Score" on a 3-D spatial grid, from 3 spatial coordinates (`x1`–`x3`)

Only polynomial regression (polynomial feature expansion + linear regression / Ridge) is used, as required.

---

## Sample Submission Notes

`sample_submission.csv` has exactly **one column: `y`** (no ID column).  
Row order in the prediction files must match the row order of the corresponding test CSVs exactly.  
No discrepancy found between `sample_submission.csv` and this instruction file.

---

## EDA Findings (Step 1)

| Dataset | Shape | Features | Target | NaNs | Duplicates |
|---|---|---|---|---|---|
| train_var1 | 1000 × 7 | x1–x6 | y | 0 | 0 |
| test_var1 | 1000 × 6 | x1–x6 | — | 0 | 0 |
| train_var2 | 1000 × 4 | x1–x3 | y | 0 | 0 |
| test_var2 | 1000 × 3 | x1–x3 | — | 0 | 0 |

- All features are pre-scaled to **[-1, 1]**. No constant columns. No outlier removal was needed.
- `sample_submission.csv` has column `y` only (no ID). Predictions are row-aligned with the test files.

---

## Environment Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.12. Pinned versions in `requirements.txt`.

---

## Reproducing Results

Run in this exact order from the **repo root**:

```bash
python src/select_degree.py
python src/train_predict.py
```

- `select_degree.py` runs the full CV sweep for both problems, saves results + plots to `results/`, and writes `results/selection_summary.csv`.
- `train_predict.py` reads `results/selection_summary.csv`, refits on the full training data, and writes the two prediction CSVs.

All results use `random_state=42` / `numpy.random.seed(42)` for reproducibility.

---

## Chosen Degrees and Models

| Problem | Degree | Model | CV Alpha | CV MSE | CV R² | Holdout MSE | Holdout R² |
|---|---|---|---|---|---|---|---|
| var1 | **5** | Ridge | 1.0 | 0.4753 | 0.9538 | 0.4035 | 0.9593 |
| var2 | **8** | Ridge | 0.01 | 0.2541 | 0.9942 | 0.2630 | 0.9951 |

**Degree selection rule:** 1-standard-error rule over 5-fold CV MSE — the smallest degree whose mean CV MSE ≤ best_mse + 1 SE (parsimony).  
**Ridge** was chosen over OLS in both cases because it strictly improved CV MSE.

---

## File Map

```
.
├── assignment.md
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── data.py              # load_train, load_test, validate/write predictions
│   ├── model.py             # build_ols_pipeline, build_ridge_pipeline, n_poly_terms
│   ├── select_degree.py     # CV sweep, OLS vs Ridge comparison, plots
│   ├── train_predict.py     # final fit on full train, write prediction CSVs
│   └── utils.py             # set_global_seed, print_df_summary
├── results/
│   ├── cv_results_var1.csv
│   ├── cv_results_var1_ols.csv
│   ├── cv_results_var2.csv
│   ├── cv_results_var2_ols.csv
│   ├── degree_curve_var1.png
│   ├── degree_curve_var2.png
│   └── selection_summary.csv
├── report/
│   ├── report.md
│   └── BT2024062_report.pdf
├── BT2024062_pred_var1.csv
└── BT2024062_pred_var2.csv
```

---

## Git Commands (do not push unless explicitly asked)

```bash
git init
git add .
git commit -m "BT2024062: polynomial regression assignment"
# To push:
# git remote add origin <your-remote-url>
# git push -u origin main
```
