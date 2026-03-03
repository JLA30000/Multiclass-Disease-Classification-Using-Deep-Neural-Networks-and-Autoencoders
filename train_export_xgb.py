import argparse
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
import xgboost as xgb

from dataset import diagnosticsDataset


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


def mean_std(xs):
    xs = np.asarray(xs, dtype=float)
    return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def can_use_xgb_cuda() -> bool:
    try:
        X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        y = np.array([0, 1, 0, 1], dtype=np.float32)
        d = xgb.DMatrix(X, label=y)
        xgb.train(
            {
                "objective": "binary:logistic",
                "tree_method": "hist",
                "device": "cuda",
                "max_depth": 1,
                "verbosity": 0,
            },
            d,
            num_boost_round=1,
            verbose_eval=False,
        )
        return True
    except Exception:
        return False


def resolve_device(requested_device: str) -> str:
    req = requested_device.strip().lower()
    if req == "auto":
        return "cuda" if can_use_xgb_cuda() else "cpu"
    if req.startswith("cuda"):
        if not can_use_xgb_cuda():
            raise RuntimeError(
                "Requested CUDA device, but CUDA training is not available in this environment."
            )
        return req
    return "cpu"


def compute_class_weights_from_train(train_labels: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(
        train_labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    N = float(len(train_labels))
    w = N / (num_classes * counts)
    return w.astype(np.float32)


def make_sample_weights(labels: np.ndarray, class_weights: np.ndarray):
    return class_weights[labels].astype(np.float32)


def eval_metrics(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        top_k_accuracy_score
    )
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    labels = np.arange(num_classes)
    k3 = min(3, num_classes)
    k5 = min(5, num_classes)
    top3 = top_k_accuracy_score(y_true, y_prob, k=k3, labels=labels)
    top5 = top_k_accuracy_score(y_true, y_prob, k=k5, labels=labels)

    return {
        "acc": acc,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "macro_f1": macro_f1,
        "top3": top3,
        "top5": top5,
    }


def train_one_seed(
    seed: int,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    class_weights: np.ndarray,
    params: dict,
    early_stopping_rounds: int,
    num_classes: int,
) -> Tuple[Dict[str, Any], xgb.Booster]:
    set_seed(seed)

    sw_train = make_sample_weights(y_train, class_weights)
    sw_val = make_sample_weights(y_val, class_weights)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sw_train)
    dval = xgb.DMatrix(X_val, label=y_val, weight=sw_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    xgb_params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "max_depth": params["max_depth"],
        "learning_rate": params["learning_rate"],
        "min_child_weight": params["min_child_weight"],
        "gamma": params["gamma"],
        "subsample": params["subsample"],
        "colsample_bytree": params["colsample_bytree"],
        "reg_lambda": params["reg_lambda"],
        "reg_alpha": params["reg_alpha"],
        "tree_method": params["tree_method"],
        "device": params["device"],
        "nthread": params["n_jobs"],
        "seed": seed,
        "verbosity": 0,
        "eval_metric": "mlogloss",
    }

    bst = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=params["n_estimators"],
        evals=[(dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )

    val_prob = bst.predict(dval)
    test_prob = bst.predict(dtest)

    val_m = eval_metrics(y_val, val_prob, num_classes)
    test_m = eval_metrics(y_test, test_prob, num_classes)

    metrics = {
        "seed": int(seed),
        "best_val_macro_f1": float(val_m["macro_f1"]),
        "test_acc": float(test_m["acc"]),
        "test_macro_p": float(test_m["macro_p"]),
        "test_macro_r": float(test_m["macro_r"]),
        "test_macro_f1": float(test_m["macro_f1"]),
        "test_top3": float(test_m["top3"]),
        "test_top5": float(test_m["top5"]),
    }

    return metrics, bst


# ----------------------------
# bundle config
# ----------------------------
@dataclass
class XGBBundleConfig:
    bundle_format: str

    data_path: str
    data_sha256: str
    indices_file: str
    indices_sha256: str

    label_column: str
    feature_cols: List[str]

    num_classes: int

    # xgb hyperparams
    n_estimators: int
    learning_rate: float
    max_depth: int
    min_child_weight: float
    gamma: float
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    reg_alpha: float
    tree_method: str
    early_stopping_rounds: int

    seeds: List[int]
    device: str


def main():
    parser = argparse.ArgumentParser(
        "Train class-weighted XGBoost + export backend bundle")

    # data
    parser.add_argument("--data_path", type=str, default="diseases.csv")
    parser.add_argument("--indices_file", type=str,
                        default="split_indices_full_80_10_10.npz")
    parser.add_argument("--label_column", type=str, default="diseases")

    # seeds / early stopping
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--early_stopping_rounds", type=int, default=50)

    # xgb params
    parser.add_argument("--n_estimators", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--min_child_weight", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample_bytree", type=float, default=0.9)
    parser.add_argument("--reg_lambda", type=float, default=1.0)
    parser.add_argument("--reg_alpha", type=float, default=0.0)
    parser.add_argument("--tree_method", type=str,
                        default="hist", choices=["hist", "approx", "exact"])
    parser.add_argument("--device", type=str, default="auto",
                        help="XGBoost device: auto | cpu | cuda")
    parser.add_argument("--n_jobs", type=int, default=8)

    # export
    parser.add_argument("--export_dir", type=str, default="export_xgb")
    parser.add_argument("--export_name", type=str, default="xgb_classifier")

    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)

    resolved_device = resolve_device(args.device)
    if resolved_device.startswith("cuda") and args.tree_method != "hist":
        print("CUDA requested; overriding tree_method to 'hist' for GPU compatibility.")
        args.tree_method = "hist"

    print(f"XGBoost device: {resolved_device} | tree_method: {args.tree_method}")
    print(f"Seeds: {seeds}")
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

    X_train = train_ds.features.numpy().astype(np.float32)
    y_train = train_ds.labels.numpy().astype(np.int64)
    X_val = val_ds.features.numpy().astype(np.float32)
    y_val = val_ds.labels.numpy().astype(np.int64)
    X_test = test_ds.features.numpy().astype(np.float32)
    y_test = test_ds.labels.numpy().astype(np.int64)

    label_names = list(train_ds.label_names)
    num_classes = len(label_names)

    print(f"Input dim:   {input_dim}")
    print(f"Num classes: {num_classes}")
    print("=" * 70)

    # ----- Class weights from TRAIN only -----
    class_w = compute_class_weights_from_train(y_train, num_classes)

    params = {
        "num_class": num_classes,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "gamma": args.gamma,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_lambda": args.reg_lambda,
        "reg_alpha": args.reg_alpha,
        "tree_method": args.tree_method,
        "device": resolved_device,
        "n_jobs": args.n_jobs,
    }

    # ----- Run seeds; pick best by validation macro-F1 -----
    t0 = time.time()
    all_metrics: List[Dict[str, Any]] = []
    best_metrics = None
    best_bst = None
    best_val_f1 = -1.0

    for s in seeds:
        metrics, bst = train_one_seed(
            seed=s,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            class_weights=class_w,
            params=params,
            early_stopping_rounds=args.early_stopping_rounds,
            num_classes=num_classes,
        )
        all_metrics.append(metrics)

        print(
            f"[seed {s}] bestValF1={metrics['best_val_macro_f1']:.4f} | "
            f"testF1={metrics['test_macro_f1']:.4f} | acc={metrics['test_acc']:.4f} | "
            f"top3={metrics['test_top3']:.4f} | top5={metrics['test_top5']:.4f}"
        )

        if metrics["best_val_macro_f1"] > best_val_f1:
            best_val_f1 = metrics["best_val_macro_f1"]
            best_metrics = metrics
            best_bst = bst

    if best_bst is None or best_metrics is None:
        raise RuntimeError(
            "No best model selected. Check training/early stopping.")

    # ----- Summary mean/std on TEST -----
    def collect(k): return [m[k] for m in all_metrics]
    summary = {}
    for key in ["test_acc", "test_macro_p", "test_macro_r", "test_macro_f1", "test_top3", "test_top5"]:
        mu, sd = mean_std(collect(key))
        summary[key] = {"mean": mu, "std": sd}

    # ----- Build config + bundle -----
    cfg = XGBBundleConfig(
        bundle_format="xgb_classifier_bundle_v1",

        data_path=args.data_path,
        data_sha256=sha256_file(args.data_path),
        indices_file=args.indices_file,
        indices_sha256=sha256_file(args.indices_file),

        label_column=args.label_column,
        feature_cols=feature_cols,

        num_classes=num_classes,

        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        tree_method=args.tree_method,
        early_stopping_rounds=args.early_stopping_rounds,

        seeds=seeds,
        device=resolved_device,
    )

    ensure_dir(args.export_dir)

    # Save XGBoost booster as raw bytes (not a PyTorch state dict)
    booster_raw = best_bst.save_raw()

    bundle = {
        "format": cfg.bundle_format,
        "config": asdict(cfg),

        # schema/decoding (backend needs these)
        "feature_cols": feature_cols,
        "label_column": args.label_column,
        "label_names": label_names,

        # XGBoost booster as raw bytes
        "xgb_booster_raw": booster_raw,

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
