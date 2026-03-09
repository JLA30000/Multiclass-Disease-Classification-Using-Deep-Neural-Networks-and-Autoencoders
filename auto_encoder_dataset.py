import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class AutoencoderDataset(Dataset):

    def __init__(self, csv_path, split='train', indices_file='split_indices.npz', transform=None, label_column='diseases', verbose=True):
        self.transform = transform
        df = pd.read_csv(csv_path)
        if verbose:
            print(f'Loaded {len(df)} total samples from {csv_path}')
        if label_column in df.columns:
            X_df = df.drop(columns=[label_column])
        else:
            X_df = df
        X_df = X_df.apply(pd.to_numeric, errors='raise')
        self.X = X_df.to_numpy(dtype=np.float32)
        indices = np.load(indices_file)
        if split == 'train':
            split_indices = indices['train_idx']
        elif split == 'val' and 'val_idx' in indices:
            split_indices = indices['val_idx']
        elif split == 'test':
            split_indices = indices['test_idx']
        else:
            raise ValueError("split must be 'train', 'val' (with val_idx present), or 'test'")
        self.X = self.X[np.asarray(split_indices, dtype=np.int64)]
        if verbose:
            print(f'Autoencoder {split} dataset: {len(self.X)} samples')
            print(f'Feature dimension: {self.X.shape[1]}')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        if self.transform:
            x = self.transform(x)
        return x
