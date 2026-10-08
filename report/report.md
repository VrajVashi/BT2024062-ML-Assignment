# Polynomial Regression Assignment Report
**Roll Number:** BT2024062

---

## 1. Problem & Data Overview

Two regression tasks are provided, each with a separate training set (1 000 samples) and a held-out test set (1 000 samples, no labels).

| Problem | Features | Target | Description |
|---|---|---|---|
| var1 | x1–x6 (6 continuous) | y ∈ [-10.1, 10.6] | Steam turbine Net Power Score |
| var2 | x1–x3 (3 continuous) | y ∈ [-30.0, 39.1] | 3D grid Thermal Anomaly Score |

**EDA findings:**
- All features are pre-scaled to **[-1, 1]**; no further normalisation was strictly necessary.
- Zero NaN values, zero duplicate rows, no constant columns in either dataset.
- No data removal was performed (datasets are described as pre-cleaned).
- `sample_submission.csv` contains one column `y`, row-aligned to the test files (no ID column).

---

## 2. Method

### Pipeline (identical structure for both problems)

```
MinMaxScaler(feature_range=(-1, 1))
  → PolynomialFeatures(degree=d, include_bias=True)
  → Ridge(alpha=a, fit_intercept=False)
```

The scaler is included inside the sklearn `Pipeline` so it is fit only on the training fold during cross-validation (no data leakage).  
`LinearRegression` (OLS, SVD-based) was evaluated as a baseline; Ridge was the final model for both problems.

### Degree definition
`PolynomialFeatures(degree=d)` uses **total degree ≤ d** across all features, matching the assignment definition exactly.

### Cross-validation protocol
- **5-fold CV** (`KFold(n_splits=5, shuffle=True, random_state=42)`) was the primary protocol.
- Metrics recorded per degree: mean train MSE, mean val MSE, val std, mean val R², number of polynomial terms.
- **Degree selection rule (1-SE / parsimony):** choose the *smallest* degree d such that mean_val_MSE(d) ≤ best_mean_val_MSE + 1 SE (where SE = std / sqrt(k) over k=5 folds). This prevents selecting a slightly-overfitted higher degree when a simpler model performs equivalently.

### Ridge regularisation
For each problem, once the OLS-optimal degree was identified, a **coarse fixed-alpha search** was run at that degree: alpha ∈ {0.001, 0.01, 0.1, 1, 10, 100, 1000} with 5-fold CV. The best alpha from this single-degree search was then used as a fixed value for the full degree sweep (not tuned per degree). This is a simplification — a finer per-degree grid would give marginal gains (e.g. alpha=3 at var1 d=5 gives CV MSE 0.4605 vs 0.4753; within noise). Ridge was used whenever it improved CV MSE over OLS.

---

## 3. Results per Problem

### 3.1 var1 — Steam Turbine Net Power Score

**Number of polynomial terms (6 features):**

| Degree | Terms | Train MSE | Val MSE | Val Std | Val R² |
|---|---|---|---|---|---|
| 1 | 7 | 9.0384 | 9.1058 | 0.575 | 0.114 |
| 2 | 28 | 3.157 | 3.400 | 0.316 | 0.669 |
| 3 | 84 | 0.817 | 1.037 | 0.146 | 0.899 |
| 4 | 210 | 0.371 | 0.686 | 0.079 | 0.933 |
| **5** | **462** | **0.146** | **0.475** | **0.065** | **0.954** |
| 6 | 924 | 0.101 | 0.564 | 0.080 | 0.945 |
| 7 | 1716 | 0.072 | 0.633 | 0.103 | 0.938 |
| 8 | 3003 | 0.052 | 0.768 | 0.172 | 0.926 |
| 9 | 5005 | 0.038 | 0.887 | 0.186 | 0.914 |
| 10 | 8008 | 0.029 | 1.028 | 0.269 | 0.900 |

*(Ridge, alpha=1.0 sweep used for this final table)*

**OLS vs Ridge at degree 4 (OLS-optimal degree):**

| Model | Alpha | CV MSE | CV R² |
|---|---|---|---|
| OLS | — | 0.7537 | 0.9267 |
| Ridge | 0.001 | 0.7535 | 0.9267 |
| Ridge | 0.010 | 0.7520 | 0.9268 |
| Ridge | 0.100 | 0.7390 | 0.9281 |
| **Ridge** | **1.0** | **0.6861** | **0.9333** |
| Ridge | 10.0 | 0.7278 | 0.9293 |
| Ridge | 100.0 | 1.4970 | 0.8544 |

Ridge(alpha=1.0) was then carried into the full degree sweep. The Ridge sweep selects degree 5 with CV MSE = 0.4753, which is 57.4% below OLS at the same degree (OLS d=5: 1.1146) and 37.0% below the best OLS at any degree (OLS d=4: 0.7537).

**Final var1 model:** degree=5, Ridge(alpha=1.0)
- 5-fold CV MSE = **0.4753**, CV R² = **0.9538**
- 80/20 holdout MSE = **0.4035**, holdout R² = **0.9593**
- Predictions: only 1.7% outside train y range (no blow-up)

### 3.2 var2 — Thermal Anomaly Score

**Number of polynomial terms (3 features):**

| Degree | Terms | Train MSE | Val MSE | Val Std | Val R² |
|---|---|---|---|---|---|
| 1 | 4 | 31.499 | 31.840 | 4.546 | 0.283 |
| 4 | 35 | 2.988 | 3.428 | 0.226 | 0.922 |
| 6 | 84 | 0.389 | 0.535 | 0.035 | 0.988 |
| 7 | 120 | 0.218 | 0.329 | 0.067 | 0.993 |
| **8** | **165** | **0.162** | **0.254** | **0.027** | **0.994** |
| 9 | 220 | 0.149 | 0.257 | 0.034 | 0.994 |
| 10 | 286 | 0.141 | 0.247 | 0.024 | 0.994 |
| 12 | 455 | 0.132 | 0.262 | 0.029 | 0.994 |
| 15+ | 816+ | — | numerical blow-up (OLS) | — | — |

OLS at degrees ≥ 15 produces catastrophic numerical blow-up (val MSE ~10^12). The Ridge sweep at d=8 recovers this completely.

**OLS vs Ridge at degree 8:**

| Model | Alpha | CV MSE | CV R² |
|---|---|---|---|
| OLS | — | 0.2663 | 0.9939 |
| Ridge | 0.001 | 0.2620 | 0.9940 |
| **Ridge** | **0.01** | **0.2541** | **0.9942** |
| Ridge | 0.1 | 0.2840 | 0.9936 |
| Ridge | 1.0 | 0.4223 | 0.9904 |

Ridge(alpha=0.01) chosen — 4.6% lower CV MSE than OLS.

**Final var2 model:** degree=8, Ridge(alpha=0.01)
- 5-fold CV MSE = **0.2541**, CV R² = **0.9942**
- 80/20 holdout MSE = **0.2630**, holdout R² = **0.9951**
- Predictions: only 0.4% outside train y range (no blow-up)

---

## 4. Numerical Considerations

| Problem | Degree | Terms | Train Samples | Conditioned? |
|---|---|---|---|---|
| var1 | 5 | 462 | 1000 | Yes — 462 < 1000, OLS determined |
| var2 | 8 | 165 | 1000 | Yes — 165 < 1000, OLS determined |

- At d=5 (var1) and d=8 (var2), the design matrix is overdetermined (more samples than terms), so OLS has a unique solution. Ridge adds a regularisation benefit regardless, confirmed by CV.
- Raw OLS at d=6+ for var1 (924–18564 terms > 1000 samples) and d=15+ for var2 become severely ill-conditioned or rank-deficient, leading to validation MSE blow-ups. Ridge completely mitigates this.
- Features were already in [-1, 1]; the `MinMaxScaler(feature_range=(-1, 1))` inside the pipeline is a no-op on the training fold but is retained for correctness.
- Both final models use `Ridge` (sklearn), which internally solves via Cholesky/SVD — no explicit normal-equation inversion.

---

## 5. Final Metrics

| Problem | Model | Degree | CV MSE | CV R² | Holdout MSE | Holdout R² |
|---|---|---|---|---|---|---|
| var1 | Ridge(α=1.0) | 5 | 0.4753 | 0.9538 | 0.4035 | 0.9593 |
| var2 | Ridge(α=0.01) | 8 | 0.2541 | 0.9942 | 0.2630 | 0.9951 |

Test-set metrics are unknown (ground truth is hidden). The holdout check on 20% of the training data is the best available independent estimate.

---

## 6. Conclusion / Limitations

- Both models achieve strong R² values (>0.95 and >0.99 respectively) using only polynomial regression + Ridge.
- The 1-SE parsimony rule correctly avoided overfitting at higher degrees.
- **Limitation:** var1 predictions extend slightly beyond the training y range (1.7%), indicating mild extrapolation in the test set — acceptable and expected for polynomial models.
- **Limitation (test distribution shift for var1):** ~50.3% of var1 test feature values sit exactly at ±1 (the boundary of the input range), versus only 31.8% in training. Test points are therefore more concentrated at the edges of the domain, where polynomial models are least reliable. The CV MSE of 0.4753 was estimated on the training distribution and may be optimistic for the hidden test set. This further supports using Ridge regularisation and a moderate degree (5) rather than higher degrees.
- **Limitation:** the true degree of the data-generating function is unknown; the chosen degrees (5 and 8) minimise cross-validated MSE and are consistent with the stated hints (≤10 and ≤20).
- Plain OLS at high degrees for var2 (d≥15) is numerically catastrophic without regularisation — the Ridge approach is essential.
