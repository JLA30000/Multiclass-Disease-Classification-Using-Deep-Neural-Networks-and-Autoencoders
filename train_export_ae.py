import argparse
import copy
import hashlib
import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    top_k_accuracy_score,
)

from dataset import diagnosticsDataset
from neural_network import NeuralNetwork
from neural_network_autoencoder import Autoencoder  # must match your AE class


# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_seeds(s: str) -> List[int]:
    s = s.replace(",", " ").strip()
    return [int(x) for x in s.split() if x.strip()]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    mu = float(arr.mean())
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mu, sd


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state_dict_any(path: str, device: torch.device) -> Tuple[Dict[str, torch.Tensor], Any]:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], ckpt
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"], ckpt
    if isinstance(ckpt, dict):
        # some people store raw state dict directly
        maybe_tensor_keys = all(isinstance(v, torch.Tensor)
                                for v in ckpt.values())
        if maybe_tensor_keys:
            return ckpt, ckpt
    return ckpt, ckpt


# ----------------------------
# Feature extraction
# ----------------------------
@torch.no_grad()
def extract_bottleneck(ae: nn.Module, raw_loader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    ae.eval()
    feats, labels = [], []
    for X, y in raw_loader:
        X = X.to(device).float()
        z = ae.encoder(X)  # [B, z_dim]
        feats.append(z.cpu())
        labels.append(y.cpu())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


# ----------------------------
# Evaluation
# ----------------------------
@torch.no_grad()
def eval_classifier(
    clf: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Dict[str, Any]:
    clf.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    all_true, all_pred, all_probs = [], [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = clf(X)
        loss = criterion(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total += bs

        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()

        all_true.append(y.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    if total == 0:
        return {
            "loss": 0.0,
            "acc": 0.0,
            "macro_p": 0.0,
            "macro_r": 0.0,
            "macro_f1": 0.0,
            "top3": 0.0,
            "top5": 0.0,
            "cm": np.zeros((num_classes, num_classes), dtype=np.int64),
        }

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_probs)

    loss_avg = total_loss / total
    acc = correct / total

    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    labels = np.arange(num_classes)
    k3 = min(3, num_classes)
    k5 = min(5, num_classes)
    top3 = top_k_accuracy_score(y_true, y_prob, k=k3, labels=labels)
    top5 = top_k_accuracy_score(y_true, y_prob, k=k5, labels=labels)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "loss": float(loss_avg),
        "acc": float(acc),
        "macro_p": float(macro_p),
        "macro_r": float(macro_r),
        "macro_f1": float(macro_f1),
        "top3": float(top3),
        "top5": float(top5),
        "cm": cm,
    }


# ----------------------------
# Config saved for backend
# ----------------------------
@dataclass
class BundleConfig:
    bundle_format: str

    # data/schema
    data_path: str
    data_sha256: str
    indices_file: str
    indices_sha256: str
    label_column: str
    feature_cols: List[str]

    # model/arch
    raw_input_dim: int
    z_dim: int
    hidden_dims: List[int]
    activation: str
    num_classes: int

    # checkpoints
    ae_ckpt_path: str
    ae_ckpt_sha256: str

    # train hyperparams
    lr: float
    batch_size: int
    epochs: int
    patience: int
    min_delta: float
    min_epochs: int

    # reproducibility
    seeds: List[int]
    device: str


def build_classifier(z_dim: int, hidden_dims: List[int], num_classes: int, activation: str, device: torch.device):
    return NeuralNetwork(
        input_dim=z_dim,
        hidden_dims=hidden_dims,
        output_dim=num_classes,
        activation=activation,
    ).to(device)


def train_one_seed(
    seed: int,
    ae: nn.Module,
    train_ds,
    val_ds,
    test_ds,
    batch_size: int,
    lr: float,
    epochs: int,
    patience: int,
    min_delta: float,
    min_epochs: int,
    hidden_dims: List[int],
    activation: str,
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    set_seed(seed)

    # deterministic bottleneck feature extraction
    raw_train = DataLoader(train_ds, batch_size=batch_size,
                           shuffle=False, num_workers=0)
    raw_val = DataLoader(val_ds, batch_size=batch_size,
                         shuffle=False, num_workers=0)
    raw_test = DataLoader(test_ds, batch_size=batch_size,
                          shuffle=False, num_workers=0)

    train_Z, train_y = extract_bottleneck(ae, raw_train, device)
    val_Z, val_y = extract_bottleneck(ae, raw_val, device)
    test_Z, test_y = extract_bottleneck(ae, raw_test, device)

    z_dim = int(train_Z.shape[1])
    num_classes = len(train_ds.label_names)

    # classifier data loaders
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TensorDataset(
        train_Z, train_y), batch_size=batch_size, shuffle=True, generator=g, num_workers=0)
    val_loader = DataLoader(TensorDataset(val_Z, val_y),
                            batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(TensorDataset(
        test_Z, test_y), batch_size=batch_size, shuffle=False, num_workers=0)

    clf = build_classifier(z_dim, hidden_dims, num_classes, activation, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clf.parameters(), lr=lr)

    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        clf.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = clf(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        val_stats = eval_classifier(
            clf, val_loader, criterion, device, num_classes)
        val_f1 = float(val_stats["macro_f1"])

        improved = (val_f1 - best_val_f1) > min_delta
        if improved:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(clf.state_dict())
            best_epoch = epoch
            patience_ctr = 0
        else:
            if epoch >= min_epochs:
                patience_ctr += 1
                if patience_ctr >= patience:
                    break

    if best_state is not None:
        clf.load_state_dict(best_state)

    test_stats = eval_classifier(
        clf, test_loader, criterion, device, num_classes)

    metrics = {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_f1),
        "test_loss": float(test_stats["loss"]),
        "test_acc": float(test_stats["acc"]),
        "test_macro_p": float(test_stats["macro_p"]),
        "test_macro_r": float(test_stats["macro_r"]),
        "test_macro_f1": float(test_stats["macro_f1"]),
        "test_top3": float(test_stats["top3"]),
        "test_top5": float(test_stats["top5"]),
    }

    # (optional) keep cm if you want it later
    metrics["_cm"] = test_stats["cm"]

    return metrics, best_state


def main():
    p = argparse.ArgumentParser(
        "Train AE bottleneck classifier + export backend bundle")

    # data
    p.add_argument("--data_path", type=str, default="filtered_diseases.csv")
    p.add_argument("--indices_file", type=str, default="split_indices.npz")
    p.add_argument("--label_column", type=str, default="diseases")

    # AE checkpoint
    p.add_argument("--ae_ckpt_path", type=str,
                   default="autoencoder_seed42_lr0.001_bs64_bestVAL0.000860_test0.000849.pth")

    # training
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seeds", type=str, default="42")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument("--min_epochs", type=int, default=10)

    # classifier arch
    p.add_argument("--hidden_dims", type=int,
                   nargs="+", default=[256, 128, 64])
    p.add_argument("--activation", type=str, default="relu")

    # export
    p.add_argument("--export_dir", type=str, default="export_model_200")
    p.add_argument("--export_name", type=str, default="ae_classifier_200")

    args = p.parse_args()
    seeds = parse_seeds(args.seeds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seeds:  {seeds}")
    print("=" * 70)

    # ----- Load full CSV to capture canonical feature order -----
    full_df = pd.read_csv(args.data_path)
    if args.label_column not in full_df.columns:
        raise ValueError(
            f"Label column '{args.label_column}' not found in CSV columns.")
    feature_cols = [c for c in full_df.columns if c != args.label_column]
    raw_input_dim = len(feature_cols)

    print(f"Raw input dim: {raw_input_dim}")
    print(f"Label column:  {args.label_column}")
    print(f"Num features:  {len(feature_cols)}")
    print("=" * 70)

    # ----- Datasets (use your existing class) -----
    train_ds = diagnosticsDataset(args.data_path, split="train",
                                  indices_file=args.indices_file, label_column=args.label_column)
    val_ds = diagnosticsDataset(args.data_path, split="val",
                                indices_file=args.indices_file, label_column=args.label_column)
    test_ds = diagnosticsDataset(args.data_path, split="test",
                                 indices_file=args.indices_file, label_column=args.label_column)

    # sanity check: dataset feature dim matches CSV feature_cols
    if int(train_ds.features.shape[1]) != raw_input_dim:
        raise RuntimeError(
            f"Feature dim mismatch: dataset has {train_ds.features.shape[1]} but CSV feature_cols has {raw_input_dim}."
        )

    label_names = list(train_ds.label_names)
    num_classes = len(label_names)

    print(f"Num classes:   {num_classes}")
    print("=" * 70)

    # ----- Load AE -----
    ae = Autoencoder(input_dim=raw_input_dim).to(device)
    ae_state, ae_meta = load_state_dict_any(args.ae_ckpt_path, device)
    ae.load_state_dict(ae_state)
    ae.eval()

    # ----- Train across seeds, keep best by validation macro-F1 -----
    all_metrics = []
    best_metrics = None
    best_state = None
    best_val_f1 = -1.0

    for s in seeds:
        metrics, state = train_one_seed(
            seed=s,
            ae=ae,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            min_epochs=args.min_epochs,
            hidden_dims=args.hidden_dims,
            activation=args.activation,
            device=device,
        )
        all_metrics.append(metrics)

        print(
            f"[seed {s}] bestValF1={metrics['best_val_macro_f1']:.4f} @epoch {metrics['best_epoch']} | "
            f"testF1={metrics['test_macro_f1']:.4f} | acc={metrics['test_acc']:.4f} | "
            f"top3={metrics['test_top3']:.4f} | top5={metrics['test_top5']:.4f}"
        )

        if state is not None and metrics["best_val_macro_f1"] > best_val_f1:
            best_val_f1 = metrics["best_val_macro_f1"]
            best_metrics = metrics
            best_state = state

    if best_state is None or best_metrics is None:
        raise RuntimeError(
            "No best model selected. Something went wrong in training/early stopping.")

    # z_dim is whatever AE encoder outputs
    with torch.no_grad():
        sample_x = train_ds[0][0].unsqueeze(0).to(device).float()
        sample_z = ae.encoder(sample_x)
        z_dim = int(sample_z.shape[1])

    # ----- Summaries -----
    def collect(k): return [m[k] for m in all_metrics]
    summary = {}
    for key in ["test_loss", "test_acc", "test_macro_p", "test_macro_r", "test_macro_f1", "test_top3", "test_top5"]:
        mu, sd = mean_std(collect(key))
        summary[key] = {"mean": mu, "std": sd}

    # ----- Build bundle config -----
    cfg = BundleConfig(
        bundle_format="ae_bottleneck_classifier_bundle_v1",

        data_path=args.data_path,
        data_sha256=sha256_file(args.data_path),

        indices_file=args.indices_file,
        indices_sha256=sha256_file(args.indices_file),

        label_column=args.label_column,
        feature_cols=feature_cols,

        raw_input_dim=raw_input_dim,
        z_dim=z_dim,
        hidden_dims=list(args.hidden_dims),
        activation=args.activation,
        num_classes=num_classes,

        ae_ckpt_path=args.ae_ckpt_path,
        ae_ckpt_sha256=sha256_file(args.ae_ckpt_path),

        lr=float(args.lr),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        patience=int(args.patience),
        min_delta=float(args.min_delta),
        min_epochs=int(args.min_epochs),

        seeds=seeds,
        device=str(device),
    )

    # ----- Export -----
    ensure_dir(args.export_dir)

    bundle = {
        "format": cfg.bundle_format,
        "config": asdict(cfg),

        # schema / decoding
        "feature_cols": feature_cols,
        "label_column": args.label_column,
        "label_names": label_names,

        # weights
        "autoencoder_state_dict": ae.state_dict(),
        "classifier_state_dict": best_state,

        # provenance / metrics
        "best_seed_metrics": {k: v for k, v in best_metrics.items() if k != "_cm"},
        "all_seed_metrics": [{k: v for k, v in m.items() if k != "_cm"} for m in all_metrics],
        "summary_mean_std": summary,

        # optional: keep AE ckpt keys for debugging
        "ae_ckpt_meta_keys": list(ae_meta.keys()) if isinstance(ae_meta, dict) else None,
    }

    bundle_path = os.path.join(
        args.export_dir, f"{args.export_name}.bundle.pt")
    torch.save(bundle, bundle_path)

    # also save a readable json alongside (nice for debugging)
    json_path = os.path.join(
        args.export_dir, f"{args.export_name}.bundle.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "format": bundle["format"],
                "config": bundle["config"],
                "label_names_count": len(label_names),
                "feature_cols_count": len(feature_cols),
                "best_seed_metrics": bundle["best_seed_metrics"],
                "summary_mean_std": bundle["summary_mean_std"],
            },
            f,
            indent=2,
        )

    print("\n=== EXPORT COMPLETE ===")
    print(f"Bundle: {bundle_path}")
    print(f"Summary JSON: {json_path}")
    print("Backend should load ONLY the .bundle.pt file.\n")


if __name__ == "__main__":
    main()
