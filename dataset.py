import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class diagnosticsDataset(Dataset):
    def __init__(self, file_path, split="train", indices_file="split_indices.npz",
                 label_column="diseases"):

        full_df = pd.read_csv(file_path)

        if label_column not in full_df.columns:
            raise ValueError(
                f"Label column '{label_column}' not found. Columns: {list(full_df.columns)}")

        # Fit label encoder ONCE (global) so mapping is consistent across splits
        all_labels = full_df[label_column].astype(str).str.strip().values
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(all_labels)

        indices = np.load(indices_file)

        if split == "train":
            split_indices = indices["train_idx"]
        elif split == "test":
            split_indices = indices["test_idx"]
        elif split == "val" and "val_idx" in indices:
            split_indices = indices["val_idx"]
        else:
            raise ValueError(
                "split must be 'train', 'test', or 'val' (with val_idx present)")

        # Apply split
        df = full_df.iloc[split_indices].reset_index(drop=True)

        df[label_column] = df[label_column].astype(str).str.strip()

        feature_cols = [c for c in df.columns if c != label_column]
        features = df[feature_cols].values.astype("float32")

        labels = self.label_encoder.transform(df[label_column].values)

        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels).long()
        self.label_names = self.label_encoder.classes_

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
