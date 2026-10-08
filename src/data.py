import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_train(variant: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(ROOT / f"BT2024062_train_{variant}.csv")
    feature_cols = [c for c in df.columns if c != "y"]
    X = df[feature_cols].astype(np.float64)
    y = df["y"].astype(np.float64)
    return X, y


def load_test(variant: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / f"BT2024062_test_{variant}.csv")
    return df.astype(np.float64)


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(ROOT / "sample_submission.csv")


def validate_predictions(pred: np.ndarray, test_df: pd.DataFrame, variant: str) -> None:
    assert len(pred) == len(test_df), (
        f"Row count mismatch: got {len(pred)}, expected {len(test_df)}"
    )
    assert not np.any(np.isnan(pred)), "NaN found in predictions"
    assert not np.any(np.isinf(pred)), "Inf found in predictions"
    fname = ROOT / f"BT2024062_pred_{variant}.csv"
    assert fname.name == f"BT2024062_pred_{variant}.csv", "Wrong filename"
    print(f"[validate] {variant} passed all assertions — {len(pred)} rows, no NaN/Inf")


def write_predictions(pred: np.ndarray, variant: str) -> None:
    out_path = ROOT / f"BT2024062_pred_{variant}.csv"
    pd.DataFrame({"y": pred}).to_csv(out_path, index=False)
    print(f"Written: {out_path}")
