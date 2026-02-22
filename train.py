import argparse
import copy
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
from neural_network import NeuralNetwork
from utils import set_seed


def parse_seeds(s: str):
    # accepts "0,1,2" or "0 1 2"
    s = s.replace(",", " ").strip()
    return [int(x) for x in s.split() if x.strip() != ""]


def aggregate_confusion_matrix(cm: np.ndarray, group_size: int) -> np.ndarray:
    n = cm.shape[0]
    g = (n + group_size - 1) // group_size
    agg = np.zeros((g, g), dtype=np.int64)
    for i in range(n):
        gi = i // group_size
        row = cm[i]
        for j in range(n):
            agg[gi, j // group_size] += row[j]
    return agg


def normalize_cm(cm: np.ndarray, mode: str):
    if mode == "none":
        return cm.astype(float)

    cm = cm.astype(float)
    if mode == "true":  # row-normalize
        row_sums = cm.sum(axis=1, keepdims=True)
        return np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
    if mode == "pred":  # col-normalize
        col_sums = cm.sum(axis=0, keepdims=True)
        return np.divide(cm, col_sums, out=np.zeros_like(cm), where=col_sums != 0)
    if mode == "all":
        total = cm.sum()
        return cm / total if total != 0 else cm
    raise ValueError("normalize must be one of: true | pred | all | none")


def plot_block_confusion(block_cm: np.ndarray, group_size: int, n_classes: int, normalize: str, out_prefix: str):
    cm_show = normalize_cm(block_cm, normalize)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(cm_show, aspect="auto")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Proportion" if normalize != "none" else "Count")

    g = block_cm.shape[0]
    tick_labels = [
        f"{k*group_size}-{min((k+1)*group_size-1, n_classes-1)}" for k in range(g)]
    ax.set_xticks(np.arange(g))
    ax.set_yticks(np.arange(g))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)

    ax.set_xlabel("Predicted class block")
    ax.set_ylabel("True class block")
    ax.set_title(
        f"Aggregated Block Confusion Matrix (block={group_size}, normalize={normalize})")

    plt.tight_layout()
    plt.savefig(out_prefix + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(out_prefix + ".jpg", dpi=300, bbox_inches="tight")
    plt.show()


def train_one_seed(
    seed: int,
    train_dataset,
    val_dataset,
    test_dataset,
    input_dim: int,
    output_dim: int,
    lr: float,
    batch_size: int,
    epochs: int,
    patience: int,
    min_delta: float,
    min_epochs: int,
    device
):
    set_seed(seed)

    # reproducible shuffle
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    model = NeuralNetwork(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128, 64, 32],
        output_dim=output_dim,
        activation="relu"
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def run_eval(loader):
        model.eval()
        total_loss = 0.0
        total = 0
        correct = 0
        all_true, all_pred, all_probs = [], [], []

        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = criterion(logits, y)

                bs = y.size(0)
                total_loss += loss.item() * bs
                total += bs

                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(logits, dim=1)

                correct += (pred == y).sum().item()

                all_true.append(y.cpu().numpy())
                all_pred.append(pred.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        all_true = np.concatenate(
            all_true) if all_true else np.array([], dtype=int)
        all_pred = np.concatenate(
            all_pred) if all_pred else np.array([], dtype=int)
        all_probs = np.concatenate(all_probs) if all_probs else np.empty(
            (0, output_dim), dtype=float)

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)

        macro_p = precision_score(
            all_true, all_pred, average="macro", zero_division=0)
        macro_r = recall_score(
            all_true, all_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(all_true, all_pred,
                            average="macro", zero_division=0)

        labels = np.arange(output_dim)
        top3 = top_k_accuracy_score(
            all_true, all_probs, k=min(3, output_dim), labels=labels)
        top5 = top_k_accuracy_score(
            all_true, all_probs, k=min(5, output_dim), labels=labels)

        cm = confusion_matrix(all_true, all_pred, labels=labels)

        return avg_loss, acc, macro_p, macro_r, macro_f1, top3, top5, cm, all_true, all_pred

    # ---- Early stopping on VAL macro F1 ----
    best_val_f1 = -1.0
    best_state = None
    patience_ctr = 0

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        val_loss, val_acc, val_p, val_r, val_f1, val_top3, val_top5, _, _, _ = run_eval(
            val_loader)

        improvement = val_f1 - best_val_f1
        if improvement > min_delta:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            if epoch + 1 >= min_epochs:
                patience_ctr += 1
                if patience_ctr >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Final TEST eval ----
    test_loss, test_acc, test_p, test_r, test_f1, test_top3, test_top5, cm, y_true, y_pred = run_eval(
        test_loader)

    metrics = {
        "loss": test_loss,
        "acc": test_acc,
        "macro_p": test_p,
        "macro_r": test_r,
        "macro_f1": test_f1,
        "top3": test_top3,
        "top5": test_top5
    }

    return metrics, cm, y_true, y_pred


def mean_std(xs):
    xs = np.asarray(xs, dtype=float)
    return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser(
        "10-seed training + aggregated confusion analysis (no model saving, no training plots)")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=0.001)
    parser.add_argument("--min_epochs", type=int, default=10)

    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")

    parser.add_argument("--group_size", type=int, default=25,
                        help="classes per block in aggregated confusion heatmap")
    parser.add_argument("--normalize", type=str, default="true",
                        choices=["true", "pred", "all", "none"])

    parser.add_argument("--top_confusions", type=int, default=5)
    parser.add_argument("--rep_classes", type=int, default=12,
                        help="how many representative diseases to print")

    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seeds: {seeds}")
    print("=" * 70)

    # ---- Dataset ----
    train_dataset = diagnosticsDataset(
        "filtered_diseases.csv", split="train", indices_file="split_indices.npz")
    val_dataset = diagnosticsDataset(
        "filtered_diseases.csv", split="val",   indices_file="split_indices.npz")
    test_dataset = diagnosticsDataset(
        "filtered_diseases.csv", split="test",  indices_file="split_indices.npz")

    input_dim = train_dataset.features.shape[1]
    output_dim = len(train_dataset.label_names)
    label_names = list(train_dataset.label_names)

    print(f"Input dim: {input_dim} | Classes: {output_dim}")
    print("=" * 70)

    # ---- Run seeds ----
    metrics_list = []
    cms = []
    all_true = []
    all_pred = []

    for s in seeds:
        m, cm, y_true, y_pred = train_one_seed(
            seed=s,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            input_dim=input_dim,
            output_dim=output_dim,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            min_epochs=args.min_epochs,
            device=device
        )

        metrics_list.append(m)
        cms.append(cm)
        all_true.append(y_true)
        all_pred.append(y_pred)

        print(f"[seed {s}] "
              f"loss={m['loss']:.4f} | acc={m['acc']:.4f} | "
              f"macroP/R/F1={m['macro_p']:.4f}/{m['macro_r']:.4f}/{m['macro_f1']:.4f} | "
              f"top3={m['top3']:.4f} | top5={m['top5']:.4f}")

    # ---- Mean ± std metrics ----
    def collect(key): return [m[key] for m in metrics_list]

    print("\n=== TEST METRICS over seeds (mean ± std) ===")
    for key, name in [
        ("loss", "Test Loss"),
        ("acc", "Test Acc"),
        ("macro_p", "Macro Precision"),
        ("macro_r", "Macro Recall"),
        ("macro_f1", "Macro F1"),
        ("top3", "Top-3 Acc"),
        ("top5", "Top-5 Acc"),
    ]:
        mu, sd = mean_std(collect(key))
        print(f"{name:16s}: {mu:.4f} ± {sd:.4f}")

    # ---- Aggregated confusion over seeds (sum of counts) ----
    cm_total = np.sum(np.stack(cms, axis=0), axis=0)

    # ---- Block heatmap ----
    block_cm = aggregate_confusion_matrix(cm_total, group_size=args.group_size)
    out_prefix = f"agg_block_confusion_g{args.group_size}_seeds{seeds[0]}-{seeds[-1]}_100"
    plot_block_confusion(block_cm, args.group_size,
                         output_dim, args.normalize, out_prefix)

    # ---- Top-K most confused pairs (off-diagonal) ----
    cm_off = cm_total.copy()
    np.fill_diagonal(cm_off, 0)
    flat = cm_off.ravel()
    topk = min(args.top_confusions, flat.size)

    idxs = np.argpartition(-flat, topk-1)[:topk]
    idxs = idxs[np.argsort(-flat[idxs])]  # sort desc

    print(
        f"\n=== Top-{topk} most commonly confused (True → Pred) across all seeds ===")
    for rank, idx in enumerate(idxs, start=1):
        i = idx // output_dim
        j = idx % output_dim
        c = int(cm_off[i, j])
        if c <= 0:
            continue
        print(
            f"{rank:2d}. {label_names[i]}  →  {label_names[j]}   (count={c})")

    # ---- Per-class performance (aggregated over seeds) ----
    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)

    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
        y_true_all, y_pred_all, labels=np.arange(output_dim), zero_division=0
    )

    # Representative set: lowest F1, median-ish F1, highest F1 (all with support>0)
    valid = per_support > 0
    f1_valid = per_f1.copy()
    f1_valid[~valid] = np.inf

    worst_idx = np.argsort(f1_valid)[: max(1, args.rep_classes // 3)]
    best_idx = np.argsort(-per_f1)[: max(1, args.rep_classes // 3)]

    # median-ish
    mids = np.argsort(per_f1)
    mids = [i for i in mids if valid[i]]
    mid_take = mids[len(mids)//2: len(mids)//2 + max(1, args.rep_classes // 3)]

    reps = []
    for arr in [worst_idx, mid_take, best_idx]:
        for i in arr:
            if valid[i] and i not in reps:
                reps.append(i)
    reps = reps[:args.rep_classes]

    print("\n=== Representative per-class performance (aggregated over seeds) ===")
    print("Disease | Support | Precision | Recall | F1")
    for i in reps:
        print(
            f"{label_names[i]} | {int(per_support[i])} | {per_p[i]:.4f} | {per_r[i]:.4f} | {per_f1[i]:.4f}")

    print("\nDone. (Saved heatmap as PNG+JPG with prefix:", out_prefix + ")")


if __name__ == "__main__":
    main()
