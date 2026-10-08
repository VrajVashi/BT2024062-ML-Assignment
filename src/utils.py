import numpy as np
import pandas as pd

RANDOM_SEED = 42


def set_global_seed():
    np.random.seed(RANDOM_SEED)


def print_df_summary(df: pd.DataFrame, name: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nHead:\n{df.head()}")
    print(f"\nDescribe:\n{df.describe()}")
    print(f"\nNull counts:\n{df.isnull().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
