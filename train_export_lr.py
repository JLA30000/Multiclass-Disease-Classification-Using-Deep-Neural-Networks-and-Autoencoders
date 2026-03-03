import argparse
import copy
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score, recall_score, f1_score, top_k_accuracy_score
)

from dataset import diagnosticsDataset
from logistic_regression_weighted import LogisticRegressionSoftmax


# ----------------------------
# helpers
# ----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def parse_seeds(s):
    if isinstance(s, int):
        return [s]
    s = str(s).replace(",", " ").strip()
    return [int(x) for x in s.split() if x.strip()]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean_std(xs):
    xs = np.asarray(xs, dtype=float)
    return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_class_weights_from_train(train_labels: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(
        train_labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    N = float(len(train_labels))
    w = N / (num_classes * counts)
    return w.astype(np.float32)


@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader, criterion, device: torch.device, num_classes: int):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    all_preds, all_true, all_probs = [], [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total += bs

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()

        all_preds.append(preds.detach().cpu().numpy())
        all_true.append(y.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)

    all_preds = np.concatenate(
        all_preds) if all_preds else np.array([], dtype=int)
    all_true = np.concatenate(
        all_true) if all_true else np.array([], dtype=int)
    all_probs = np.concatenate(all_probs) if all_probs else np.empty(
        (0, num_classes), dtype=float)

    macro_p = precision_score(
        all_true, all_preds, average="macro", zero_division=0) if len(all_true) else 0.0
    macro_r = recall_score(all_true, all_preds, average="macro",
                           zero_division=0) if len(all_true) else 0.0
    macro_f1 = f1_score(all_true, all_preds, average="macro",
                        zero_division=0) if len(all_true) else 0.0

    labels = np.arange(num_classes)
    k3 = min(3, num_classes)
    k5 = min(5, num_classes)
    top3 = top_k_accuracy_score(
        all_true, all_probs, k=k3, labels=labels) if len(all_true) else 0.0
    top5 = top_k_accuracy_score(
        all_true, all_probs, k=k5, labels=labels) if len(all_true) else 0.0

    return float(avg_loss), float(acc), float(macro_p), float(macro_r), float(macro_f1), float(top3), float(top5)


def train_one_seed(
    seed: int,
    train_ds, val_ds, test_ds,
    batch_size: int,
    lr: float,
    epochs: int,
    patience: int,
    min_delta: float,
    min_epochs: int,
    weight_decay: float,
    weights_tensor: torch.Tensor,
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    set_seed(seed)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=g, num_workers=0)
    val_loader = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    input_dim = int(train_ds.features.shape[1])
    num_classes = len(train_ds.label_names)

    model = LogisticRegressionSoftmax(
        input_dim=input_dim, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        val_loss, val_acc, val_p, val_r, val_f1, val_top3, val_top5 = run_eval(
            model, val_loader, criterion, device, num_classes
        )

        improved = (val_f1 - best_val_f1) > min_delta
        if improved:
            best_val_f1 = float(val_f1)
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = int(epoch)
            patience_ctr = 0
        else:
            if epoch >= min_epochs:
                patience_ctr += 1
                if patience_ctr >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_p, test_r, test_f1, test_top3, test_top5 = run_eval(
        model, test_loader, criterion, device, num_classes
    )

    metrics = {
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_f1),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_macro_p": float(test_p),
        "test_macro_r": float(test_r),
        "test_macro_f1": float(test_f1),
        "test_top3": float(test_top3),
        "test_top5": float(test_top5),
    }

    return metrics, best_state


# ----------------------------
# bundle config
# ----------------------------
@dataclass
class LRBundleConfig:
    bundle_format: str

    data_path: str
    data_sha256: str
    indices_file: str
    indices_sha256: str

    label_column: str
    feature_cols: List[str]

    input_dim: int
    num_classes: int

    lr: float
    weight_decay: float
    batch_size: int
    epochs: int
    patience: int
    min_delta: float
    min_epochs: int

    seeds: List[int]
    device: str


def main():
    parser = argparse.ArgumentParser(
        "Train class-weighted Logistic Regression + export backend bundle")

    # data
    parser.add_argument("--data_path", type=str, default="diseases.csv")
    parser.add_argument("--indices_file", type=str,
                        default="split_indices_full_80_10_10.npz")
    parser.add_argument("--label_column", type=str, default="diseases")

    # training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--seeds", type=str, default="0")

    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--min_epochs", type=int, default=10)

    # export
    parser.add_argument("--export_dir", type=str, default="export_lr")
    parser.add_argument("--export_name", type=str, default="lr_classifier")

    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seeds:  {seeds}")
    print("=" * 70)

    # ----- Capture canonical feature order from CSV -----
    full_df = pd.read_csv(args.data_path)
    if args.label_column not in full_df.columns:
        raise ValueError(
            f"Label column '{args.label_column}' not found in CSV.")
    feature_cols = [c for c in full_df.columns if c != args.label_column]
    input_dim = len(feature_cols)

    # ----- Load datasets -----
    train_ds = diagnosticsDataset(args.data_path, split="train",
                                  indices_file=args.indices_file, label_column=args.label_column)
    val_ds = diagnosticsDataset(args.data_path, split="val",
                                indices_file=args.indices_file, label_column=args.label_column)
    test_ds = diagnosticsDataset(args.data_path, split="test",
                                 indices_file=args.indices_file, label_column=args.label_column)

    if int(train_ds.features.shape[1]) != input_dim:
        raise RuntimeError(
            f"Dataset feature dim {train_ds.features.shape[1]} != CSV feature_cols {input_dim}")

    label_names = list(train_ds.label_names)
    num_classes = len(label_names)

    print(f"Input dim:   {input_dim}")
    print(f"Num classes: {num_classes}")
    print("=" * 70)

    # ----- Compute class weights from TRAIN labels only -----
    w = compute_class_weights_from_train(train_ds.labels.numpy(), num_classes)
    weights_tensor = torch.tensor(w, dtype=torch.float32).to(device)

    # ----- Run seeds; pick best by validation macro-F1 -----
    t0 = time.time()
    all_metrics: List[Dict[str, Any]] = []
    best_metrics = None
    best_state = None
    best_val_f1 = -1.0

    for s in seeds:
        metrics, state = train_one_seed(
            seed=s,
            train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            min_epochs=args.min_epochs,
            weight_decay=args.weight_decay,
            weights_tensor=weights_tensor,
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
            "No best model selected. Check training/early stopping.")

    # ----- Summary mean/std on TEST -----
    def collect(k): return [m[k] for m in all_metrics]
    summary = {}
    for key in ["test_loss", "test_acc", "test_macro_p", "test_macro_r", "test_macro_f1", "test_top3", "test_top5"]:
        mu, sd = mean_std(collect(key))
        summary[key] = {"mean": mu, "std": sd}

    # ----- Build config + bundle -----
    cfg = LRBundleConfig(
        bundle_format="lr_classifier_bundle_v1",

        data_path=args.data_path,
        data_sha256=sha256_file(args.data_path),
        indices_file=args.indices_file,
        indices_sha256=sha256_file(args.indices_file),

        label_column=args.label_column,
        feature_cols=feature_cols,

        input_dim=input_dim,
        num_classes=num_classes,

        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        patience=int(args.patience),
        min_delta=float(args.min_delta),
        min_epochs=int(args.min_epochs),

        seeds=seeds,
        device=str(device),
    )

    ensure_dir(args.export_dir)

    bundle = {
        "format": cfg.bundle_format,
        "config": asdict(cfg),

        # schema/decoding (backend needs these)
        "feature_cols": feature_cols,
        "label_column": args.label_column,
        "label_names": label_names,

        # weights
        "classifier_state_dict": best_state,

        # training-time class weights (for reproducibility, not needed at inference)
        "class_weights": w,

        # metrics/provenance
        "best_seed_metrics": best_metrics,
        "all_seed_metrics": all_metrics,
        "summary_mean_std": summary,
    }

    bundle_path = os.path.join(
        args.export_dir, f"{args.export_name}.bundle.pt")
    torch.save(bundle, bundle_path)

    # readable sidecar JSON
    json_path = os.path.join(
        args.export_dir, f"{args.export_name}.bundle.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "format": bundle["format"],
                "config": bundle["config"],
                "feature_cols_count": len(feature_cols),
                "label_names_count": len(label_names),
                "best_seed_metrics": bundle["best_seed_metrics"],
                "summary_mean_std": bundle["summary_mean_std"],
            },
            f,
            indent=2
        )

    print("\n=== EXPORT COMPLETE ===")
    print(f"Bundle: {bundle_path}")
    print(f"Summary JSON: {json_path}")
    print(f"Total runtime: {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
