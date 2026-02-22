# make_splits.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.10   # final held-out test
VAL_SIZE = 0.10   # validation carved out of the remaining train pool
CSV_FILE = "filtered_diseases.csv"
SAVE_FILE = "split_indices.npz"
LABEL_COL = "diseases"  # change if your label column name differs

df = pd.read_csv(CSV_FILE)

if LABEL_COL not in df.columns:
    raise ValueError(f"'{LABEL_COL}' not found. Columns: {list(df.columns)}")

y = df[LABEL_COL].astype(str).str.strip().values
idx = np.arange(len(y))

# 1) Train pool vs Test (stratified)
idx_train_pool, idx_test = train_test_split(
    idx,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=SEED,
)

# 2) Train vs Val (stratified) from the train pool
y_train_pool = y[idx_train_pool]
val_frac_of_train_pool = VAL_SIZE / \
    (1.0 - TEST_SIZE)  # so VAL_SIZE is of total

idx_train, idx_val = train_test_split(
    idx_train_pool,
    test_size=val_frac_of_train_pool,
    stratify=y_train_pool,
    random_state=SEED,
)

# Sanity checks
idx_train = np.array(idx_train, dtype=np.int64)
idx_val = np.array(idx_val, dtype=np.int64)
idx_test = np.array(idx_test, dtype=np.int64)

assert len(set(idx_train) & set(idx_val)) == 0
assert len(set(idx_train) & set(idx_test)) == 0
assert len(set(idx_val) & set(idx_test)) == 0
assert len(idx_train) + len(idx_val) + len(idx_test) == len(idx)

np.savez(
    SAVE_FILE,
    train_idx=idx_train,
    val_idx=idx_val,
    test_idx=idx_test,
    seed=SEED,
    test_size=TEST_SIZE,
    val_size=VAL_SIZE,
    label_col=LABEL_COL,
)

print(f"Saved splits to {SAVE_FILE}")
print(f"Train: {len(idx_train)} ({len(idx_train)/len(idx):.1%})")
print(f"Val  : {len(idx_val)} ({len(idx_val)/len(idx):.1%})")
print(f"Test : {len(idx_test)} ({len(idx_test)/len(idx):.1%})")
