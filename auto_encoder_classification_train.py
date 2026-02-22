import argparse
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, top_k_accuracy_score,
    precision_recall_fscore_support
)

from dataset import diagnosticsDataset
from neural_network import NeuralNetwork
from neural_network_autoencoder import Autoencoder  # must match your AE class


# ----------------- HARDCODED AUTOENCODER CHECKPOINT -----------------
AE_CKPT_PATH = "autoencoder_seed42_lr0.001_bs64_bestVAL0.001385_test0.001240.pth"


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
def extract_bottleneck_features(ae: nn.Module, raw_loader: DataLoader, device: torch.device):
    ae.eval()
    feats, labs = [], []
    for X, y in raw_loader:
        X = X.to(device).float()
        z = ae.encoder(X)  # [B, z_dim]
        feats.append(z.cpu())
        labs.append(y.cpu())
    return torch.cat(feats, dim=0), torch.cat(labs, dim=0)


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
        all_true, all_preds, average="macro", zero_division=0)
    macro_r = recall_score(all_true, all_preds,
                           average="macro", zero_division=0)
    macro_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)

    labels = np.arange(num_classes)
    k3 = min(3, num_classes)
    k5 = min(5, num_classes)
    top3 = top_k_accuracy_score(
        all_true, all_probs, k=k3, labels=labels) if len(all_true) else 0.0
    top5 = top_k_accuracy_score(
        all_true, all_probs, k=k5, labels=labels) if len(all_true) else 0.0

    cm = confusion_matrix(all_true, all_preds, labels=labels)

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
    plt.show()


def mean_std(xs):
    xs = np.asarray(xs, dtype=float)
    return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0


def train_one_seed(
    seed: int,
    ae: nn.Module,
    train_ds, val_ds, test_ds,
    batch_size: int,
    lr: float,
    epochs: int,
    patience: int,
    min_delta: float,
    min_epochs: int,
    hidden_dims,
    device: torch.device
):
    set_seed(seed)

    # ---- Raw loaders for feature extraction (no shuffle) ----
    raw_train = DataLoader(train_ds, batch_size=batch_size,
                           shuffle=False, num_workers=0)
    raw_val = DataLoader(val_ds,   batch_size=batch_size,
                         shuffle=False, num_workers=0)
    raw_test = DataLoader(test_ds,  batch_size=batch_size,
                          shuffle=False, num_workers=0)

    # ---- Extract bottleneck features once per seed (AE fixed) ----
    train_Z, train_y = extract_bottleneck_features(ae, raw_train, device)
    val_Z,   val_y = extract_bottleneck_features(ae, raw_val, device)
    test_Z,  test_y = extract_bottleneck_features(ae, raw_test, device)

    z_dim = train_Z.shape[1]
    num_classes = len(train_ds.label_names)

    # ---- Classifier loaders (shuffle train with seed) ----
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TensorDataset(
        train_Z, train_y), batch_size=batch_size, shuffle=True, generator=g, num_workers=0)
    val_loader = DataLoader(TensorDataset(val_Z, val_y),
                            batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(TensorDataset(
        test_Z, test_y),   batch_size=batch_size, shuffle=False, num_workers=0)

    clf = NeuralNetwork(
        input_dim=z_dim,
        hidden_dims=hidden_dims,
        output_dim=num_classes,
        activation="relu"
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clf.parameters(), lr=lr)

    best_val_f1 = -1.0
    best_state = None
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

        # early stopping on VAL macro F1
        val_loss, val_acc, val_p, val_r, val_f1, val_top3, val_top5, _, _, _ = run_eval(
            clf, val_loader, criterion, device, num_classes
        )

        if (val_f1 - best_val_f1) > min_delta:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(clf.state_dict())
            patience_ctr = 0
        else:
            if epoch >= min_epochs:
                patience_ctr += 1
                if patience_ctr >= patience:
                    break

    if best_state is not None:
        clf.load_state_dict(best_state)

    test_loss, test_acc, test_p, test_r, test_f1, test_top3, test_top5, cm, y_true, y_pred = run_eval(
        clf, test_loader, criterion, device, num_classes
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
    return metrics, cm, y_true, y_pred


def main():
    parser = argparse.ArgumentParser(
        "AE bottleneck classifier — 10 seeds + aggregated confusion (no saving/plots)")
    parser.add_argument("--data_path", type=str,
                        default="filtered_diseases.csv")
    parser.add_argument("--indices_file", type=str,
                        default="split_indices.npz")

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)

    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")

    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--min_epochs", type=int, default=10)

    parser.add_argument("--hidden_dims", type=int,
                        nargs="+", default=[256, 128, 64])

    parser.add_argument("--group_size", type=int, default=25)
    parser.add_argument("--normalize", type=str, default="true",
                        choices=["true", "pred", "all", "none"])

    parser.add_argument("--top_confusions", type=int, default=5)
    parser.add_argument("--rep_classes", type=int, default=12)

    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)

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
    print("=" * 70)

    # ---- Load autoencoder (fixed) ----
    print("Loading trained autoencoder...")
    print(f"Autoencoder checkpoint: {AE_CKPT_PATH}")

    ae = Autoencoder(input_dim=input_dim).to(device)
    ckpt = torch.load(AE_CKPT_PATH, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(
        ckpt, dict) and "model_state_dict" in ckpt else ckpt
    ae.load_state_dict(state)
    ae.eval()

    # ---- Run seeds ----
    metrics_list = []
    cms = []
    all_true = []
    all_pred = []

    for s in seeds:
        m, cm, y_true, y_pred = train_one_seed(
            seed=s,
            ae=ae,
            train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            min_epochs=args.min_epochs,
            hidden_dims=args.hidden_dims,
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

    # ---- Mean ± std ----
    def collect(k): return [m[k] for m in metrics_list]

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

    # ---- Aggregated confusion (sum across seeds) ----
    cm_total = np.sum(np.stack(cms, axis=0), axis=0)

    # ---- Block heatmap ----
    block_cm = aggregate_blocks(cm_total, args.group_size)
    out_prefix = f"AEclf_block_confusion_g{args.group_size}_seeds{seeds[0]}-{seeds[-1]}_100"
    plot_block_heatmap(block_cm, args.group_size,
                       num_classes, args.normalize, out_prefix)

    # ---- Top-K confusion pairs ----
    cm_off = cm_total.copy()
    np.fill_diagonal(cm_off, 0)
    flat = cm_off.ravel()
    topk = min(args.top_confusions, flat.size)

    idxs = np.argpartition(-flat, topk - 1)[:topk]
    idxs = idxs[np.argsort(-flat[idxs])]

    print(
        f"\n=== Top-{topk} most commonly confused (True → Pred) across all seeds ===")
    for r, idx in enumerate(idxs, start=1):
        i = idx // num_classes
        j = idx % num_classes
        c = int(cm_off[i, j])
        if c <= 0:
            continue
        print(f"{r:2d}. {label_names[i]}  →  {label_names[j]}   (count={c})")

    # ---- Representative per-class P/R/F1 ----
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
                reps.append(i)
    reps = reps[:args.rep_classes]

    print("\n=== Representative per-class performance (aggregated over seeds) ===")
    print("Disease | Support | Precision | Recall | F1")
    for i in reps:
        print(
            f"{label_names[i]} | {int(per_support[i])} | {per_p[i]:.4f} | {per_r[i]:.4f} | {per_f1[i]:.4f}")

    print("\nDone.")
    print(f"Saved heatmap: {out_prefix}.png and {out_prefix}.jpg")


if __name__ == "__main__":
    main()
