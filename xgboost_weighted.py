import argparse
import random
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, top_k_accuracy_score, precision_recall_fscore_support
)

from dataset import diagnosticsDataset

# XGBoost
try:
    import xgboost as xgb
except ImportError as e:
    raise ImportError(
        "xgboost is not installed. Install with:\n"
        "  pip install xgboost\n"
        "or (conda):\n"
        "  conda install -c conda-forge xgboost\n"
    ) from e


# ----------------- helpers -----------------
def parse_seeds(s: str):
    s = s.replace(",", " ").strip()
    return [int(x) for x in s.split() if x.strip() != ""]


def can_use_xgb_cuda() -> bool:
    """
    Checks whether XGBoost can successfully run a tiny CUDA training job.
    """
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


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


def make_sample_weights(labels: np.ndarray, class_weights: np.ndarray):
    return class_weights[labels].astype(np.float32)


def aggregate_blocks(cm: np.ndarray, group_size: int) -> np.ndarray:
    n = cm.shape[0]
    g = (n + group_size - 1) // group_size
    out = np.zeros((g, g), dtype=np.int64)
    for i in range(n):
        gi = i // group_size
        for j in range(n):
            out[gi, j // group_size] += cm[i, j]
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
        f"XGBoost Aggregated Block Confusion (block={group_size}, normalize={normalize})")

    plt.tight_layout()
    plt.savefig(out_prefix + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(out_prefix + ".jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)


def eval_metrics(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int):
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

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "acc": acc,
        "macro_p": macro_p,
        "macro_r": macro_r,
        "macro_f1": macro_f1,
        "top3": top3,
        "top5": top5,
        "cm": cm,
        "y_pred": y_pred
    }


def train_one_seed(
    seed: int,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    class_weights: np.ndarray,
    params: dict,
    early_stopping_rounds: int
):
    set_seed(seed)
    num_classes = int(params["num_class"])

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

    return val_m, test_m


def main():
    parser = argparse.ArgumentParser(
        "XGBoost Multiclass Disease Classifier (class-weighted, multi-seed)"
    )

    parser.add_argument("--data_path", type=str, default="diseases.csv")
    parser.add_argument("--indices_file", type=str,
                        default="split_indices_full_80_10_10.npz")

    # seeds / early stopping
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--early_stopping_rounds", type=int, default=50)

    # xgb params
    parser.add_argument("--n_estimators", type=int, default=4000)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--min_child_weight", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--subsample", type=float, default=0.75)
    parser.add_argument("--colsample_bytree", type=float, default=0.9)
    parser.add_argument("--reg_lambda", type=float, default=0.5)  # L2
    parser.add_argument("--reg_alpha", type=float, default=0.0)   # L1
    parser.add_argument("--tree_method", type=str,
                        default="hist", choices=["hist", "approx", "exact"])
    parser.add_argument("--device", type=str, default="auto",
                        help="XGBoost device: auto | cpu | cuda (or cuda:0)")
    parser.add_argument("--n_jobs", type=int, default=8)

    # confusion / reporting
    parser.add_argument("--group_size", type=int, default=25)
    parser.add_argument("--normalize", type=str, default="true",
                        choices=["true", "pred", "all", "none"])
    parser.add_argument("--top_confusions", type=int, default=5)
    parser.add_argument("--rep_classes", type=int, default=12)

    parser.add_argument("--out_prefix", type=str, default=None,
                        help="Prefix for saved heatmap files; if None, auto-named")
    parser.add_argument("--no_plots", action="store_true",
                        help="Disable saving heatmaps (useful for fast grid search).")

    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    if not seeds:
        raise ValueError("No seeds parsed. Pass --seeds like '0' or '0,1,2'.")
    if args.group_size <= 0:
        raise ValueError("--group_size must be > 0.")
    if args.top_confusions < 0:
        raise ValueError("--top_confusions must be >= 0.")

    # ---- Load data ----
    train_ds = diagnosticsDataset(
        args.data_path, split="train", indices_file=args.indices_file)
    val_ds = diagnosticsDataset(
        args.data_path, split="val",   indices_file=args.indices_file)
    test_ds = diagnosticsDataset(
        args.data_path, split="test",  indices_file=args.indices_file)

    X_train = train_ds.features.numpy().astype(np.float32)
    y_train = train_ds.labels.numpy().astype(np.int64)
    X_val = val_ds.features.numpy().astype(np.float32)
    y_val = val_ds.labels.numpy().astype(np.int64)
    X_test = test_ds.features.numpy().astype(np.float32)
    y_test = test_ds.labels.numpy().astype(np.int64)

    label_names = list(train_ds.label_names)
    num_classes = len(label_names)

    print(
        f"Classes: {num_classes} | Train/Val/Test: {len(y_train)}/{len(y_val)}/{len(y_test)}")
    print(f"Seeds: {seeds}")
    print("=" * 70)

    resolved_device = resolve_device(args.device)
    if resolved_device.startswith("cuda") and args.tree_method != "hist":
        print("CUDA requested; overriding tree_method to 'hist' for GPU compatibility.")
        args.tree_method = "hist"
    print(f"XGBoost device: {resolved_device} | tree_method: {args.tree_method}")

    # ---- Class weights from TRAIN only ----
    class_w = compute_class_weights_from_train(y_train, num_classes)

    # ---- Params ----
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
        "n_jobs": args.n_jobs
    }

    # ---- Run seeds ----
    t0 = time.time()

    val_metrics_list = []
    test_metrics_list = []
    cms = []
    all_true = []
    all_pred = []

    for s in seeds:
        val_m, test_m = train_one_seed(
            seed=s,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            class_weights=class_w,
            params=params,
            early_stopping_rounds=args.early_stopping_rounds
        )

        val_metrics_list.append(val_m)
        test_metrics_list.append(test_m)

        cms.append(test_m["cm"])
        all_true.append(y_test)
        all_pred.append(test_m["y_pred"])

        print(f"[seed {s}] "
              f"VAL macroF1={val_m['macro_f1']:.6f} | "
              f"TEST acc={test_m['acc']:.6f} macroF1={test_m['macro_f1']:.6f} top3={test_m['top3']:.6f} top5={test_m['top5']:.6f}")

    # ---- Grid-search parseable metric line (mean VAL macroF1 across seeds) ----
    val_macro_f1s = [m["macro_f1"] for m in val_metrics_list]
    val_accs = [m["acc"] for m in val_metrics_list]
    val_top3s = [m["top3"] for m in val_metrics_list]
    val_top5s = [m["top5"] for m in val_metrics_list]

    val_mu, val_sd = mean_std(val_macro_f1s)
    val_acc_mu, _ = mean_std(val_accs)
    val_top3_mu, _ = mean_std(val_top3s)
    val_top5_mu, _ = mean_std(val_top5s)

    print(f"GRID_METRIC val_macro_f1 {val_mu:.6f}")
    print(f"GRID_METRIC val_macro_f1_std {val_sd:.6f}")
    print(f"GRID_METRIC val_acc {val_acc_mu:.6f}")
    print(f"GRID_METRIC val_top3 {val_top3_mu:.6f}")
    print(f"GRID_METRIC val_top5 {val_top5_mu:.6f}")

    # ---- Mean +/- std (TEST) ----
    def collect(k): return [m[k] for m in test_metrics_list]

    print("\n=== TEST METRICS over seeds (mean +/- std) ===")
    for key, name in [
        ("acc", "Test Acc"),
        ("macro_p", "Macro Precision"),
        ("macro_r", "Macro Recall"),
        ("macro_f1", "Macro F1"),
        ("top3", "Top-3 Acc"),
        ("top5", "Top-5 Acc"),
    ]:
        mu, sd = mean_std(collect(key))
        print(f"{name:16s}: {mu:.6f} +/- {sd:.6f}")

    # ---- Aggregated confusion (sum across seeds) ----
    cm_total = np.sum(np.stack(cms, axis=0), axis=0)

    if args.out_prefix is None:
        out_prefix = (
            f"XGB_block_g{args.group_size}_lr{args.learning_rate}_md{args.max_depth}_"
            f"ne{args.n_estimators}_sub{args.subsample}_col{args.colsample_bytree}_"
            f"lam{args.reg_lambda}_alp{args.reg_alpha}"
        )
    else:
        out_prefix = args.out_prefix

    if not args.no_plots:
        block_cm = aggregate_blocks(cm_total, args.group_size)
        plot_block_heatmap(block_cm, args.group_size,
                           num_classes, args.normalize, out_prefix)
        print(f"\nSaved heatmap: {out_prefix}.png and {out_prefix}.jpg")
    else:
        print("\n(no_plots set) Skipping heatmap saving.")

    # ---- Confusion statistics / top confusions ----
    total = int(cm_total.sum())
    diag = int(np.trace(cm_total))
    off = total - diag
    diag_frac = diag / total if total > 0 else 0.0
    print("\n=== Aggregated confusion statistics (summed over seeds) ===")
    print(f"Total test examples (summed across seeds): {total}")
    print(f"Correct (diagonal) count: {diag} ({diag_frac:.4%})")
    print(f"Errors (off-diagonal) count: {off} ({1.0 - diag_frac:.4%})")

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
        print(f"{label_names[i]} | {int(per_support[i])} | {per_class_acc[i]:.6f} | "
              f"{per_p[i]:.6f} | {per_r[i]:.6f} | {per_f1[i]:.6f}")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

