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
from neural_network import NeuralNetwork
from neural_network_autoencoder import Autoencoder
from auto_encoder_classification_train_full import EncoderClassifier


# ----------------------------
# helpers
# ----------------------------
AE_CKPT_PATH_DEFAULT = "runs_autoencoder/autoencoder_seed0_lr0.0001_bs64_z128_bestVAL0.000248_test0.000235.pth"


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


def compute_class_weights_from_train(train_labels, num_classes: int) -> np.ndarray:
    if isinstance(train_labels, torch.Tensor):
        train_labels = train_labels.numpy()
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
    ae: nn.Module,
    train_ds, val_ds, test_ds,
    batch_size: int,
    lr: float,
    encoder_lr: float,
    epochs: int,
    patience: int,
    min_delta: float,
    min_epochs: int,
    hidden_dims: List[int],
    activation: str,
    weights_tensor: torch.Tensor,
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    set_seed(seed)

    num_classes = len(train_ds.label_names)
    z_dim = ae.encoder[-1].out_features

    # Deep-copy encoder so each seed starts from pretrained weights
    encoder = copy.deepcopy(ae.encoder)

    clf_head = NeuralNetwork(
        input_dim=z_dim,
        hidden_dims=hidden_dims,
        output_dim=num_classes,
        activation=activation,
    )

    model = EncoderClassifier(encoder, clf_head).to(device)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=g, num_workers=0)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam([
        {"params": model.encoder.parameters(), "lr": encoder_lr},
        {"params": model.classifier.parameters(), "lr": lr},
    ])

    best_val_f1 = -1.0
    best_encoder_state = None
    best_clf_state = None
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
            best_encoder_state = copy.deepcopy(model.encoder.state_dict())
            best_clf_state = copy.deepcopy(model.classifier.state_dict())
            best_epoch = int(epoch)
            patience_ctr = 0
        else:
            if epoch >= min_epochs:
                patience_ctr += 1
                if patience_ctr >= patience:
                    break

    if best_encoder_state is not None:
        model.encoder.load_state_dict(best_encoder_state)
        model.classifier.load_state_dict(best_clf_state)

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

    return metrics, best_encoder_state, best_clf_state


# ----------------------------
# bundle config
# ----------------------------
@dataclass
class AEClfFullBundleConfig:
    bundle_format: str

    data_path: str
    data_sha256: str
    indices_file: str
    indices_sha256: str

    label_column: str
    feature_cols: List[str]

    raw_input_dim: int
    latent_dim: int
    ae_hidden_dims: List[int]
    z_dim: int
    hidden_dims: List[int]
    activation: str
    num_classes: int

    ae_ckpt_path: str
    ae_ckpt_sha256: str

    lr: float
    encoder_lr: float
    batch_size: int
    epochs: int
    patience: int
    min_delta: float
    min_epochs: int

    seeds: List[int]
    device: str


def main():
    parser = argparse.ArgumentParser(
        "Train AE end-to-end fine-tuned classifier + export backend bundle")

    # data
    parser.add_argument("--data_path", type=str, default="diseases.csv")
    parser.add_argument("--indices_file", type=str,
                        default="split_indices_full_80_10_10.npz")
    parser.add_argument("--label_column", type=str, default="diseases")

    # training
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)

    parser.add_argument("--seeds", type=str, default="0")

    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--min_epochs", type=int, default=10)

    # classifier arch
    parser.add_argument("--hidden_dims", type=int,
                        nargs="+", default=[256, 128, 64])
    parser.add_argument("--activation", type=str, default="relu")

    # autoencoder
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--ae_hidden_dims", type=int,
                        nargs="+", default=[256, 128])
    parser.add_argument("--ae_checkpoint", type=str, default=AE_CKPT_PATH_DEFAULT)

    # export
    parser.add_argument("--export_dir", type=str, default="export_ae_clf")
    parser.add_argument("--export_name", type=str, default="ae_clf_full")

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
    raw_input_dim = len(feature_cols)

    # ----- Load datasets -----
    train_ds = diagnosticsDataset(args.data_path, split="train",
                                  indices_file=args.indices_file, label_column=args.label_column)
    val_ds = diagnosticsDataset(args.data_path, split="val",
                                indices_file=args.indices_file, label_column=args.label_column)
    test_ds = diagnosticsDataset(args.data_path, split="test",
                                 indices_file=args.indices_file, label_column=args.label_column)

    if int(train_ds.features.shape[1]) != raw_input_dim:
        raise RuntimeError(
            f"Dataset feature dim {train_ds.features.shape[1]} != CSV feature_cols {raw_input_dim}")

    label_names = list(train_ds.label_names)
    num_classes = len(label_names)

    print(f"Raw input dim: {raw_input_dim}")
    print(f"Num classes:   {num_classes}")
    print("=" * 70)

    # ----- Load pretrained autoencoder -----
    print(f"Loading pretrained autoencoder: {args.ae_checkpoint}")
    ae = Autoencoder(
        input_dim=raw_input_dim,
        latent_dim=args.latent_dim,
        hidden_dims=tuple(args.ae_hidden_dims),
    ).to(device)
    ckpt = torch.load(args.ae_checkpoint, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(
        ckpt, dict) and "model_state_dict" in ckpt else ckpt
    ae.load_state_dict(state)
    print(f"Autoencoder loaded (latent_dim={args.latent_dim})")

    z_dim = ae.encoder[-1].out_features

    # ----- Compute class weights from TRAIN labels only -----
    w = compute_class_weights_from_train(train_ds.labels, num_classes)
    weights_tensor = torch.tensor(w, dtype=torch.float32).to(device)

    # ----- Run seeds; pick best by validation macro-F1 -----
    t0 = time.time()
    all_metrics: List[Dict[str, Any]] = []
    best_metrics = None
    best_encoder_state = None
    best_clf_state = None
    best_val_f1 = -1.0

    for s in seeds:
        metrics, enc_state, clf_state = train_one_seed(
            seed=s,
            ae=ae,
            train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
            batch_size=args.batch_size,
            lr=args.lr,
            encoder_lr=args.encoder_lr,
            epochs=args.epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            min_epochs=args.min_epochs,
            hidden_dims=list(args.hidden_dims),
            activation=args.activation,
            weights_tensor=weights_tensor,
            device=device,
        )
        all_metrics.append(metrics)

        print(
            f"[seed {s}] bestValF1={metrics['best_val_macro_f1']:.4f} @epoch {metrics['best_epoch']} | "
            f"testF1={metrics['test_macro_f1']:.4f} | acc={metrics['test_acc']:.4f} | "
            f"top3={metrics['test_top3']:.4f} | top5={metrics['test_top5']:.4f}"
        )

        if enc_state is not None and metrics["best_val_macro_f1"] > best_val_f1:
            best_val_f1 = metrics["best_val_macro_f1"]
            best_metrics = metrics
            best_encoder_state = enc_state
            best_clf_state = clf_state

    if best_encoder_state is None or best_clf_state is None or best_metrics is None:
        raise RuntimeError(
            "No best model selected. Check training/early stopping.")

    # ----- Summary mean/std on TEST -----
    def collect(k): return [m[k] for m in all_metrics]
    summary = {}
    for key in ["test_loss", "test_acc", "test_macro_p", "test_macro_r", "test_macro_f1", "test_top3", "test_top5"]:
        mu, sd = mean_std(collect(key))
        summary[key] = {"mean": mu, "std": sd}

    # ----- Build config + bundle -----
    cfg = AEClfFullBundleConfig(
        bundle_format="ae_clf_full_bundle_v1",

        data_path=args.data_path,
        data_sha256=sha256_file(args.data_path),
        indices_file=args.indices_file,
        indices_sha256=sha256_file(args.indices_file),

        label_column=args.label_column,
        feature_cols=feature_cols,

        raw_input_dim=raw_input_dim,
        latent_dim=args.latent_dim,
        ae_hidden_dims=list(args.ae_hidden_dims),
        z_dim=z_dim,
        hidden_dims=list(args.hidden_dims),
        activation=args.activation,
        num_classes=num_classes,

        ae_ckpt_path=args.ae_checkpoint,
        ae_ckpt_sha256=sha256_file(args.ae_checkpoint),

        lr=float(args.lr),
        encoder_lr=float(args.encoder_lr),
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

        # weights: fine-tuned encoder + classifier head
        "encoder_state_dict": best_encoder_state,
        "classifier_state_dict": best_clf_state,

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
