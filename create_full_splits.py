import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
CSV_PATH = 'diseases.csv'
OUT_FILE = 'split_indices_full_80_10_10.npz'
MIN_COUNT = 10
df = pd.read_csv(CSV_PATH)
label_col = df.columns[0]
labels_raw = df[label_col].astype(str)
counts = labels_raw.value_counts()
keep_classes = counts[counts >= MIN_COUNT].index
keep_mask = labels_raw.isin(keep_classes)
nsc = (counts < MIN_COUNT).sum()
nrs = counts[counts < MIN_COUNT].sum()
print(f'Classes with < {MIN_COUNT} samples: {nsc}')
print(f'Total samples removed: {nrs}')
df_kept = df[keep_mask].reset_index(drop=False)
orig_idx = df_kept['index'].to_numpy()
labels_kept = df_kept[label_col].astype('category')
y = labels_kept.cat.codes.to_numpy()
X = df_kept.drop(columns=['index']).drop(columns=[label_col]).to_numpy()
print(f'Label column: {label_col}')
print(f'Original rows: {len(df)}')
print(f'Kept rows:     {len(df_kept)}')
print(f'Classes kept:  {labels_kept.nunique()} (min count >= {MIN_COUNT})')
X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(X, y, np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(X_temp, y_temp, idx_temp, test_size=0.5, stratify=y_temp, random_state=SEED)
train_idx = orig_idx[idx_train]
val_idx = orig_idx[idx_val]
test_idx = orig_idx[idx_test]
np.savez(OUT_FILE, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
print('Saved:', OUT_FILE)
print('Train:', len(train_idx), 'Val:', len(val_idx), 'Test:', len(test_idx))
