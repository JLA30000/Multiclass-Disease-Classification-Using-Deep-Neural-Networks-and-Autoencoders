# logistic_regression_weighted.py
import argparse
import copy
import random
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, top_k_accuracy_score,
    precision_recall_fscore_support
)

from dataset import diagnosticsDataset


# ----------------- helpers -----------------
def parse_seeds(s: str):
    s = s.replace(",", " ").strip()
    return [int(x) for x in s.split() if x.strip() != ""]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader, criterion, device: torch.device, num_classes: int):
    """
    Return:
      avg_loss, acc, macroP, macroR, macroF1, top3, top5, cm, all_true, all_preds
    """
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

        all_preds.append(preds.cpu().numpy())
        all_true.append(y.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

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

    cm = confusion_matrix(all_true, all_preds, labels=labels) if len(
        all_true) else np.zeros((num_classes, num_classes), dtype=int)

    return avg_loss, acc, macro_p, macro_r, macro_f1, top3, top5, cm, all_true, all_preds


def aggregate_blocks(cm: np.ndarray, group_size: int) -> np.ndarray:
    n = cm.shape[0]
    g = (n + group_size - 1) // group_size
    out = np.zeros((g, g), dtype=np.int64)
    for i in range(n):
        gi = i // group_size
        row = cm[i]
        for j in range(n):
            out[gi, j // group_size] += row[j]
    return out


def normalize_cm(cm: np.ndarray, mode: str):
    if mode == "none":
        return cm.astype(float)
    cm = cm.astype(float)
    if mode == "true":
        rs = cm.sum(axis=1, keepdims=True)
        return np.divide(cm, rs, out=np.zeros_like(cm), where=rs != 0)
    if mode == "pred":
        cs = cm.sum(axis=0, keepdims=True)
        return np.divide(cm, cs, out=np.zeros_like(cm), where=cs != 0)
    if mode == "all":
        total = cm.sum()
        return cm / total if total != 0 else cm
    raise ValueError("normalize must be: true | pred | all | none")


def plot_block_heatmap(block_cm: np.ndarray, group_size: int, n_classes: int, normalize: str, out_prefix: str):
    show = normalize_cm(block_cm, normalize)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(show, aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Proportion" if normalize != "none" else "Count")

    g = block_cm.shape[0]
    ticks = [
        f"{k*group_size}-{min((k+1)*group_size-1, n_classes-1)}" for k in range(g)]
    ax.set_xticks(np.arange(g))
    ax.set_yticks(np.arange(g))
    ax.set_xticklabels(ticks, rotation=45, ha="right")
    ax.set_yticklabels(ticks)

    ax.set_xlabel("Predicted class block")
    ax.set_ylabel("True class block")
    ax.set_title(
        f"Aggregated Block Confusion Matrix (block={group_size}, normalize={normalize})")

    plt.tight_layout()
    plt.savefig(out_prefix + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(out_prefix + ".jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)


def mean_std(xs):
    xs = np.asarray(xs, dtype=float)
    return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0


def compute_class_weights_from_train(train_labels: np.ndarray, num_classes: int):
    """
    weight_c = N / (C * count_c) computed ONLY on TRAIN split labels.
    """
    counts = np.bincount(
        train_labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    N = float(len(train_labels))
    w = N / (num_classes * counts)
    return w.astype(np.float32)


# ----------------- Logistic Regression model -----------------
class LogisticRegressionSoftmax(nn.Module):
    """
    Multiclass logistic regression:
      logits = X W + b
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


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
    device: torch.device
):
    set_seed(seed)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=g, num_workers=0)
    val_loader = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    input_dim = train_ds.features.shape[1]
    num_classes = len(train_ds.label_names)

    model = LogisticRegressionSoftmax(
        input_dim=input_dim, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    best_val_f1 = -1.0
    best_state = None
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

        # early stopping on VAL macro F1
        val_loss, val_acc, val_p, val_r, val_f1, val_top3, val_top5, _, _, _ = run_eval(
            model, val_loader, criterion, device, num_classes
        )

        if (val_f1 - best_val_f1) > min_delta:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            if epoch >= min_epochs:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(
                        f"[seed {seed}] Early stopping (no improvement for {patience} epochs)")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, val_acc, val_p, val_r, val_f1, val_top3, val_top5, _, _, _ = run_eval(
        model, val_loader, criterion, device, num_classes
    )

    val_metrics = {
        "loss": val_loss,
        "acc": val_acc,
        "macro_p": val_p,
        "macro_r": val_r,
        "macro_f1": val_f1,
        "top3": val_top3,
        "top5": val_top5
    }

    test_loss, test_acc, test_p, test_r, test_f1, test_top3, test_top5, cm, y_true, y_pred = run_eval(
        model, test_loader, criterion, device, num_classes
    )

    metrics = {
        "loss": test_loss,
        "acc": test_acc,
        "macro_p": test_p,
        "macro_r": test_r,
        "macro_f1": test_f1,
        "top3": test_top3,
        "top5": test_top5
    }

    return val_metrics, metrics, cm, y_true, y_pred


def main():
    parser = argparse.ArgumentParser(
        "Class-weighted Logistic Regression (softmax) - multi-seed + aggregated confusion"
    )
    parser.add_argument("--data_path", type=str, default="diseases.csv")
    parser.add_argument("--indices_file", type=str,
                        default="split_indices_full_80_10_10.npz")

    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6,
                        help="L2 regularization (Adam weight_decay).")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")

    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--min_epochs", type=int, default=10)

    # confusion + reporting
    parser.add_argument("--group_size", type=int, default=25)
    parser.add_argument("--normalize", type=str, default="true",
                        choices=["true", "pred", "all", "none"])
    parser.add_argument("--top_confusions", type=int, default=5)
    parser.add_argument("--rep_classes", type=int, default=12)

    # output prefix
    parser.add_argument("--out_prefix", type=str, default=None,
                        help="prefix for saved heatmap files; if None, auto-named")

    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    if not seeds:
        raise ValueError("No seeds parsed. Pass --seeds like '0' or '0,1,2'.")
    if args.group_size <= 0:
        raise ValueError("--group_size must be > 0.")
    if args.top_confusions < 0:
        raise ValueError("--top_confusions must be >= 0.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seeds: {seeds}")
    print("=" * 70)

    # ---- Load datasets ----
    train_ds = diagnosticsDataset(
        args.data_path, split="train", indices_file=args.indices_file)
    val_ds = diagnosticsDataset(
        args.data_path, split="val",   indices_file=args.indices_file)
    test_ds = diagnosticsDataset(
        args.data_path, split="test",  indices_file=args.indices_file)

    input_dim = train_ds.features.shape[1]
    num_classes = len(train_ds.label_names)
    label_names = list(train_ds.label_names)

    print(f"Input dim (raw): {input_dim}")
    print(f"Classes: {num_classes}")
    print(f"Random baseline acc: {1.0 / num_classes:.4%}")
    print("=" * 70)

    # ---- Compute class weights from TRAIN labels only ----
    w = compute_class_weights_from_train(train_ds.labels, num_classes)
    weights_tensor = torch.tensor(w, dtype=torch.float32).to(device)

    # ---- Run seeds ----
    t0 = time.time()

    val_metrics_list = []
    test_metrics_list = []
    cms = []
    all_true = []
    all_pred = []

    for s in seeds:
        val_m, test_m, cm, y_true, y_pred = train_one_seed(
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
            device=device
        )

        val_metrics_list.append(val_m)
        test_metrics_list.append(test_m)
        cms.append(cm)
        all_true.append(y_true)
        all_pred.append(y_pred)

        print(f"[seed {s}] "
              f"VAL macroF1={val_m['macro_f1']:.4f} | "
              f"TEST acc={test_m['acc']:.4f} | "
              f"macroP/R/F1={test_m['macro_p']:.4f}/{test_m['macro_r']:.4f}/{test_m['macro_f1']:.4f} | "
              f"top3={test_m['top3']:.4f} | top5={test_m['top5']:.4f}")

    def collect_val(k): return [m[k] for m in val_metrics_list]
    def collect_test(k): return [m[k] for m in test_metrics_list]

    val_mu_f1, val_sd_f1 = mean_std(collect_val("macro_f1"))
    val_mu_acc, _ = mean_std(collect_val("acc"))
    val_mu_top3, _ = mean_std(collect_val("top3"))
    val_mu_top5, _ = mean_std(collect_val("top5"))

    print(f"\nGRID_METRIC val_macro_f1 {val_mu_f1:.6f}")
    print(f"GRID_METRIC val_macro_f1_std {val_sd_f1:.6f}")
    print(f"GRID_METRIC val_acc {val_mu_acc:.6f}")
    print(f"GRID_METRIC val_top3 {val_mu_top3:.6f}")
    print(f"GRID_METRIC val_top5 {val_mu_top5:.6f}")

    print("\n=== TEST METRICS over seeds (mean +/- std) ===")
    for key, name in [
        ("loss", "Test Loss"),
        ("acc", "Test Acc"),
        ("macro_p", "Macro Precision"),
        ("macro_r", "Macro Recall"),
        ("macro_f1", "Macro F1"),
        ("top3", "Top-3 Acc"),
        ("top5", "Top-5 Acc"),
    ]:
        mu, sd = mean_std(collect_test(key))
        print(f"{name:16s}: {mu:.4f} +/- {sd:.4f}")

    # ---- Aggregated confusion (sum across seeds) ----
    cm_total = np.sum(np.stack(cms, axis=0), axis=0)

    # ---- Block heatmap ----
    block_cm = aggregate_blocks(cm_total, args.group_size)
    if args.out_prefix is None:
        out_prefix = f"LRW_block_confusion_g{args.group_size}_seeds{seeds[0]}-{seeds[-1]}"
    else:
        out_prefix = args.out_prefix

    plot_block_heatmap(block_cm, args.group_size,
                       num_classes, args.normalize, out_prefix)

    # ---- Confusion statistics (aggregate) ----
    total = int(cm_total.sum())
    diag = int(np.trace(cm_total))
    off = total - diag
    diag_frac = (diag / total) if total > 0 else 0.0

    print("\n=== Aggregated confusion statistics (summed over seeds) ===")
    print(f"Total test examples (summed across seeds): {total}")
    print(f"Correct (diagonal) count: {diag} ({diag_frac:.4%})")
    print(f"Errors (off-diagonal) count: {off} ({1.0 - diag_frac:.4%})")

    # ---- Top-K confusion pairs (True -> Pred) ----
    cm_off = cm_total.copy()
    np.fill_diagonal(cm_off, 0)
    flat = cm_off.ravel()
    topk = min(args.top_confusions, flat.size)

    if topk > 0:
        idxs = np.argpartition(-flat, topk - 1)[:topk]
        idxs = idxs[np.argsort(-flat[idxs])]
    else:
        idxs = np.array([], dtype=int)

    print(
        f"\n=== Top-{topk} most commonly confused (True -> Pred) across all seeds ===")
    for r, idx in enumerate(idxs, start=1):
        i = idx // num_classes
        j = idx % num_classes
        c = int(cm_off[i, j])
        if c <= 0:
            continue
        print(f"{r:2d}. {label_names[i]}  ->  {label_names[j]}   (count={c})")

    # ---- Representative per-class performance (aggregated over seeds) ----
    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)

    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
        y_true_all, y_pred_all, labels=np.arange(num_classes), zero_division=0
    )

    valid = per_support > 0
    f1_for_sort = per_f1.copy()
    f1_for_sort[~valid] = np.inf

    worst = np.argsort(f1_for_sort)[: max(1, args.rep_classes // 3)]
    best = np.argsort(-per_f1)[: max(1, args.rep_classes // 3)]

    mids = np.argsort(per_f1)
    mids = [i for i in mids if valid[i]]
    mid_take = mids[len(mids)//2: len(mids)//2 + max(1, args.rep_classes // 3)]

    reps = []
    for arr in [worst, mid_take, best]:
        for i in arr:
            if valid[i] and i not in reps:
                reps.append(int(i))
    reps = reps[:args.rep_classes]

    row_sums = cm_total.sum(axis=1)
    per_class_acc = np.divide(
        np.diag(cm_total),
        row_sums,
        out=np.zeros_like(row_sums, dtype=float),
        where=row_sums != 0
    )

    print("\n=== Representative per-class performance (aggregated over seeds) ===")
    print("Disease | Support | Accuracy | Precision | Recall | F1")
    for i in reps:
        print(f"{label_names[i]} | {int(per_support[i])} | {per_class_acc[i]:.4f} | "
              f"{per_p[i]:.4f} | {per_r[i]:.4f} | {per_f1[i]:.4f}")

    # ---- Machine-readable summary for grid search parsing ----
    best_val_mu, best_val_sd = val_mu_f1, val_sd_f1
    test_f1_mu, test_f1_sd = mean_std([m["macro_f1"] for m in test_metrics_list])

    # The .bat will parse THIS line to pick best hyperparams:
    print(f"best_val_macro_f1_mean={best_val_mu:.6f}")
    print(f"best_val_macro_f1_std={best_val_sd:.6f}")
    print(f"test_macro_f1_mean={test_f1_mu:.6f}")
    print(f"test_macro_f1_std={test_f1_sd:.6f}")

    print("\nDone.")
    print(f"Saved heatmap: {out_prefix}.png and {out_prefix}.jpg")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

