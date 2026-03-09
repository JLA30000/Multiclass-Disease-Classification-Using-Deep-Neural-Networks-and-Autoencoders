import copy
import glob
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, top_k_accuracy_score, precision_recall_fscore_support

KAGGLE_DATASET_NAME = 'ae-clf-data'
INPUT_DIR = f'/kaggle/input/{KAGGLE_DATASET_NAME}'
OUTPUT_DIR = '/kaggle/working'
DATA_PATH = os.path.join(INPUT_DIR, 'diseases.csv')
INDICES_FILE = os.path.join(INPUT_DIR, 'split_indices_full_80_10_10.npz')
AE_CKPT_GLOB = os.path.join(INPUT_DIR, 'autoencoder_seed0_lr0.0001_bs64_z64_bestVAL*.pth')
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
LR = 0.0001
ENCODER_LR = 0.0001
BATCH_SIZE = 64
EPOCHS = 50
PATIENCE = 10
MIN_DELTA = 0.0001
MIN_EPOCHS = 10
HIDDEN_DIMS = [256, 128, 64]
LATENT_DIM = 64
GROUP_SIZE = 25
NORMALIZE = 'true'
TOP_CONFUSIONS = 5
REP_CLASSES = 12


class diagnosticsDataset(Dataset):

    def __init__(self, file_path, split='train', indices_file='split_indices.npz', label_column='diseases'):
        full_df = pd.read_csv(file_path)
        if label_column not in full_df.columns:
            raise ValueError(f"Label column '{label_column}' not found. Columns: {list(full_df.columns)}")
        indices = np.load(indices_file)
        all_split_idx = np.concatenate([indices[k] for k in indices.files])
        asl = full_df[label_column].astype(str).str.strip().iloc[all_split_idx].values
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(asl)
        if split == 'train':
            split_indices = indices['train_idx']
        elif split == 'test':
            split_indices = indices['test_idx']
        elif split == 'val' and 'val_idx' in indices:
            split_indices = indices['val_idx']
        else:
            raise ValueError("split must be 'train', 'test', or 'val' (with val_idx present)")
        df = full_df.iloc[split_indices].reset_index(drop=True)
        df[label_column] = df[label_column].astype(str).str.strip()
        feature_cols = [c for c in df.columns if c != label_column]
        features = df[feature_cols].values.astype('float32')
        labels = self.label_encoder.transform(df[label_column].values)
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels).long()
        self.label_names = self.label_encoder.classes_

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return (self.features[idx], self.labels[idx])


class NeuralNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=10, activation='relu'):
        super(NeuralNetwork, self).__init__()
        activations = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(), 'leakyrelu': nn.LeakyReLU()}
        act_fn = activations.get(activation.lower(), nn.ReLU())
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act_fn)
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Autoencoder(nn.Module):

    def __init__(self, input_dim, latent_dim=64, hidden_dims=(256, 128), dropout=0.0):
        super().__init__()
        h1, h2 = hidden_dims
        enc_layers = [nn.Linear(input_dim, h1), nn.ReLU()]
        if dropout > 0:
            enc_layers.append(nn.Dropout(dropout))
        enc_layers += [nn.Linear(h1, h2), nn.ReLU()]
        if dropout > 0:
            enc_layers.append(nn.Dropout(dropout))
        enc_layers += [nn.Linear(h2, latent_dim)]
        self.encoder = nn.Sequential(*enc_layers)
        dec_layers = [nn.Linear(latent_dim, h2), nn.ReLU()]
        if dropout > 0:
            dec_layers.append(nn.Dropout(dropout))
        dec_layers += [nn.Linear(h2, h1), nn.ReLU()]
        if dropout > 0:
            dec_layers.append(nn.Dropout(dropout))
        dec_layers += [nn.Linear(h1, input_dim), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    @torch.no_grad()
    def encode(self, x):
        return self.encoder(x)


class EncoderClassifier(nn.Module):

    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()


def run_eval(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    all_preds, all_true, all_probs = ([], [], [])
    for X, y in loader:
        X, y = (X.to(device), y.to(device))
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
    all_preds = np.concatenate(all_preds) if all_preds else np.array([], dtype=int)
    all_true = np.concatenate(all_true) if all_true else np.array([], dtype=int)
    all_probs = np.concatenate(all_probs) if all_probs else np.empty((0, num_classes), dtype=float)
    macro_p = precision_score(all_true, all_preds, average='macro', zero_division=0) if len(all_true) else 0.0
    macro_r = recall_score(all_true, all_preds, average='macro', zero_division=0) if len(all_true) else 0.0
    macro_f1 = f1_score(all_true, all_preds, average='macro', zero_division=0) if len(all_true) else 0.0
    labels = np.arange(num_classes)
    k3 = min(3, num_classes)
    k5 = min(5, num_classes)
    top3 = top_k_accuracy_score(all_true, all_probs, k=k3, labels=labels) if len(all_true) else 0.0
    top5 = top_k_accuracy_score(all_true, all_probs, k=k5, labels=labels) if len(all_true) else 0.0
    cm = confusion_matrix(all_true, all_preds, labels=labels) if len(all_true) else np.zeros((num_classes, num_classes), dtype=int)
    return (avg_loss, acc, macro_p, macro_r, macro_f1, top3, top5, cm, all_true, all_preds)


def compute_class_weights_from_train(train_labels, num_classes):
    if isinstance(train_labels, torch.Tensor):
        train_labels = train_labels.numpy()
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    N = float(len(train_labels))
    w = N / (num_classes * counts)
    return w.astype(np.float32)


def mean_std(xs):
    xs = np.asarray(xs, dtype=float)
    return (float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0)


def aggregate_blocks(cm, group_size):
    n = cm.shape[0]
    g = (n + group_size - 1) // group_size
    out = np.zeros((g, g), dtype=np.int64)
    for i in range(n):
        gi = i // group_size
        row = cm[i]
        for j in range(n):
            out[gi, j // group_size] += row[j]
    return out


def normalize_cm(cm, mode):
    if mode == 'none':
        return cm.astype(float)
    cm = cm.astype(float)
    if mode == 'true':
        rs = cm.sum(axis=1, keepdims=True)
        return np.divide(cm, rs, out=np.zeros_like(cm), where=rs != 0)
    if mode == 'pred':
        cs = cm.sum(axis=0, keepdims=True)
        return np.divide(cm, cs, out=np.zeros_like(cm), where=cs != 0)
    if mode == 'all':
        total = cm.sum()
        return cm / total if total != 0 else cm
    raise ValueError('normalize must be: true | pred | all | none')


def plot_block_heatmap(block_cm, group_size, n_classes, normalize, out_prefix, *, tms=None, confusion_stats=None, top_confusions=None, rep_diseases=None):
    from matplotlib.gridspec import GridSpec
    show = normalize_cm(block_cm, normalize)
    has_extras = any((x is not None for x in [tms, top_confusions, rep_diseases]))
    if not has_extras:
        fig, ax = plt.subplots(figsize=(8.5, 7.5))
        ax_cm = ax
    else:
        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1.1, 1], width_ratios=[1.1, 1], hspace=0.35, wspace=0.3)
        ax_cm = fig.add_subplot(gs[0, 0])
        ax_metrics = fig.add_subplot(gs[0, 1])
        ax_confusions = fig.add_subplot(gs[1, 0])
        ax_rep = fig.add_subplot(gs[1, 1])
    im = ax_cm.imshow(show, aspect='auto')
    cbar = plt.colorbar(im, ax=ax_cm)
    cbar.set_label('Proportion' if normalize != 'none' else 'Count')
    g = block_cm.shape[0]
    ticks = [f'{k * group_size}-{min((k + 1) * group_size - 1, n_classes - 1)}' for k in range(g)]
    ax_cm.set_xticks(np.arange(g))
    ax_cm.set_yticks(np.arange(g))
    ax_cm.set_xticklabels(ticks, rotation=45, ha='right', fontsize=6)
    ax_cm.set_yticklabels(ticks, fontsize=6)
    ax_cm.set_xlabel('Predicted class block')
    ax_cm.set_ylabel('True class block')
    ax_cm.set_title(f'Aggregated Block Confusion Matrix (block={group_size}, normalize={normalize})')
    if not has_extras:
        plt.tight_layout()
        plt.savefig(out_prefix + '.png', dpi=300, bbox_inches='tight')
        plt.savefig(out_prefix + '.jpg', dpi=300, bbox_inches='tight')
        plt.close(fig)
        return
    ax_metrics.axis('off')
    if tms:
        lines = ['Test Metrics (mean ± std)\n']
        for key, name in [('acc', 'Accuracy'), ('macro_p', 'Macro Precision'), ('macro_r', 'Macro Recall'), ('macro_f1', 'Macro F1'), ('top3', 'Top-3 Accuracy'), ('top5', 'Top-5 Accuracy')]:
            entry = tms.get(key)
            if entry:
                lines.append(f"  {name:18s}  {entry['mean']:.4f} ± {entry['std']:.4f}")
        if confusion_stats:
            lines.append('')
            lines.append(f"  Total examples:  {confusion_stats['total']}")
            lines.append(f"  Correct:  {confusion_stats['correct']}  ({confusion_stats['correct_pct']:.2%})")
            lines.append(f"  Errors:   {confusion_stats['errors']}  ({confusion_stats['error_pct']:.2%})")
        ax_metrics.text(0.05, 0.95, '\n'.join(lines), transform=ax_metrics.transAxes, fontsize=10, fontfamily='monospace', verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))
    ax_metrics.set_title('Evaluation Metrics', fontsize=11, fontweight='bold')
    ax_confusions.axis('off')
    if top_confusions:
        lines = ['Top Confused Pairs (True → Pred)\n']
        for rank, (true_name, pred_name, count) in enumerate(top_confusions, 1):
            tn = true_name[:30] + ('..' if len(true_name) > 30 else '')
            pn = pred_name[:30] + ('..' if len(pred_name) > 30 else '')
            lines.append(f'  {rank}. {tn}  →  {pn}  ({count})')
        ax_confusions.text(0.05, 0.95, '\n'.join(lines), transform=ax_confusions.transAxes, fontsize=9, fontfamily='monospace', verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcyan', alpha=0.8))
    ax_confusions.set_title('Top Confusions', fontsize=11, fontweight='bold')
    ax_rep.axis('off')
    if rep_diseases:
        header = f"  {'Disease':<28s} {'Sup':>5s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}"
        lines = ['Representative Per-Class Performance\n', header, '  ' + '-' * 60]
        for name, support, acc, prec, rec, f1 in rep_diseases:
            dname = name[:27] + ('..' if len(name) > 27 else '')
            lines.append(f'  {dname:<28s} {support:>5d} {acc:>6.3f} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f}')
        ax_rep.text(0.05, 0.95, '\n'.join(lines), transform=ax_rep.transAxes, fontsize=8, fontfamily='monospace', verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='honeydew', alpha=0.8))
    ax_rep.set_title('Representative Diseases', fontsize=11, fontweight='bold')
    plt.savefig(out_prefix + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(out_prefix + '.jpg', dpi=300, bbox_inches='tight')
    plt.close(fig)


def train_one_seed(seed, ae, train_ds, val_ds, test_ds, batch_size, lr, encoder_lr, epochs, patience, min_delta, min_epochs, hidden_dims, weights_tensor, device):
    set_seed(seed)
    num_classes = len(train_ds.label_names)
    z_dim = ae.encoder[-1].out_features
    encoder = copy.deepcopy(ae.encoder)
    clf_head = NeuralNetwork(input_dim=z_dim, hidden_dims=hidden_dims, output_dim=num_classes, activation='relu')
    model = EncoderClassifier(encoder, clf_head).to(device)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam([{'params': model.encoder.parameters(), 'lr': encoder_lr}, {'params': model.classifier.parameters(), 'lr': lr}])
    best_val_f1 = -1.0
    best_state = None
    patience_ctr = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for X, y in train_loader:
            X, y = (X.to(device), y.to(device))
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        val_loss, val_acc, val_p, val_r, val_f1, val_top3, val_top5, _, _, _ = run_eval(model, val_loader, criterion, device, num_classes)
        if val_f1 - best_val_f1 > min_delta:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        elif epoch >= min_epochs:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f'[seed {seed}] Early stopping (no improvement for {patience} epochs)')
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_loss, val_acc, val_p, val_r, val_f1, val_top3, val_top5, _, _, _ = run_eval(model, val_loader, criterion, device, num_classes)
    val_metrics = {'loss': val_loss, 'acc': val_acc, 'macro_p': val_p, 'macro_r': val_r, 'macro_f1': val_f1, 'top3': val_top3, 'top5': val_top5}
    test_loss, test_acc, test_p, test_r, test_f1, test_top3, test_top5, cm, y_true, y_pred = run_eval(model, test_loader, criterion, device, num_classes)
    test_metrics = {'loss': test_loss, 'acc': test_acc, 'macro_p': test_p, 'macro_r': test_r, 'macro_f1': test_f1, 'top3': test_top3, 'top5': test_top5}
    return (val_metrics, test_metrics, cm, y_true, y_pred)


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Seeds: {SEEDS}')
    print('=' * 70)
    for f in [DATA_PATH, INDICES_FILE]:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Missing: {f}\nMake sure your Kaggle dataset is named '{KAGGLE_DATASET_NAME}' and contains the required files. Check INPUT_DIR: {INPUT_DIR}")
    ae_candidates = sorted(glob.glob(AE_CKPT_GLOB))
    if not ae_candidates:
        raise FileNotFoundError(f"No AE checkpoint matched: {AE_CKPT_GLOB}\nUpload the .pth file to your Kaggle dataset '{KAGGLE_DATASET_NAME}'.")
    ae_ckpt_path = ae_candidates[0]
    train_ds = diagnosticsDataset(DATA_PATH, split='train', indices_file=INDICES_FILE)
    val_ds = diagnosticsDataset(DATA_PATH, split='val', indices_file=INDICES_FILE)
    test_ds = diagnosticsDataset(DATA_PATH, split='test', indices_file=INDICES_FILE)
    input_dim = train_ds.features.shape[1]
    num_classes = len(train_ds.label_names)
    label_names = list(train_ds.label_names)
    print(f'Input dim (raw): {input_dim}')
    print(f'Classes: {num_classes}')
    print(f'Random baseline acc: {1.0 / num_classes:.4%}')
    print('=' * 70)
    print(f'Loading pretrained autoencoder: {ae_ckpt_path}')
    ae = Autoencoder(input_dim=input_dim, latent_dim=LATENT_DIM).to(device)
    ckpt = torch.load(ae_ckpt_path, map_location=device)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    ae.load_state_dict(state)
    print(f'Autoencoder loaded (latent_dim={LATENT_DIM})')
    print(f'Encoder LR: {ENCODER_LR}  |  Classifier head LR: {LR}')
    w = compute_class_weights_from_train(train_ds.labels, num_classes)
    weights_tensor = torch.tensor(w, dtype=torch.float32).to(device)
    t0 = time.time()
    vml, tml, cms, all_true, all_pred = ([], [], [], [], [])
    for s in SEEDS:
        val_m, test_m, cm, y_true, y_pred = train_one_seed(seed=s, ae=ae, train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, batch_size=BATCH_SIZE, lr=LR, encoder_lr=ENCODER_LR, epochs=EPOCHS, patience=PATIENCE, min_delta=MIN_DELTA, min_epochs=MIN_EPOCHS, hidden_dims=HIDDEN_DIMS, weights_tensor=weights_tensor, device=device)
        vml.append(val_m)
        tml.append(test_m)
        cms.append(cm)
        all_true.append(y_true)
        all_pred.append(y_pred)
        print(f"[seed {s}] VAL macroF1={val_m['macro_f1']:.4f} | TEST acc={test_m['acc']:.4f} | macroP/R/F1={test_m['macro_p']:.4f}/{test_m['macro_r']:.4f}/{test_m['macro_f1']:.4f} | top3={test_m['top3']:.4f} | top5={test_m['top5']:.4f}")

    def collect_val(k):
        return [m[k] for m in vml]

    def collect_test(k):
        return [m[k] for m in tml]
    val_mu_f1, val_sd_f1 = mean_std(collect_val('macro_f1'))
    val_mu_acc, _ = mean_std(collect_val('acc'))
    val_mu_top3, _ = mean_std(collect_val('top3'))
    val_mu_top5, _ = mean_std(collect_val('top5'))
    print(f'\nGRID_METRIC val_macro_f1 {val_mu_f1:.6f}')
    print(f'GRID_METRIC val_macro_f1_std {val_sd_f1:.6f}')
    print(f'GRID_METRIC val_acc {val_mu_acc:.6f}')
    print(f'GRID_METRIC val_top3 {val_mu_top3:.6f}')
    print(f'GRID_METRIC val_top5 {val_mu_top5:.6f}')
    tms = {}
    print('\n=== TEST METRICS over seeds (mean ± std) ===')
    for key, name in [('loss', 'Test Loss'), ('acc', 'Test Acc'), ('macro_p', 'Macro Precision'), ('macro_r', 'Macro Recall'), ('macro_f1', 'Macro F1'), ('top3', 'Top-3 Acc'), ('top5', 'Top-5 Acc')]:
        mu, sd = mean_std(collect_test(key))
        tms[key] = {'mean': mu, 'std': sd}
        print(f'{name:16s}: {mu:.4f} ± {sd:.4f}')
    cm_total = np.sum(np.stack(cms, axis=0), axis=0)
    total = int(cm_total.sum())
    diag = int(np.trace(cm_total))
    off = total - diag
    diag_frac = diag / total if total > 0 else 0.0
    confusion_stats = {'total': total, 'correct': diag, 'errors': off, 'correct_pct': diag_frac, 'error_pct': 1.0 - diag_frac}
    print(f'\n=== Aggregated confusion statistics (summed over seeds) ===')
    print(f'Total test examples (summed across seeds): {total}')
    print(f'Correct (diagonal) count: {diag} ({diag_frac:.4%})')
    print(f'Errors (off-diagonal) count: {off} ({1.0 - diag_frac:.4%})')
    cm_off = cm_total.copy()
    np.fill_diagonal(cm_off, 0)
    flat = cm_off.ravel()
    topk = min(TOP_CONFUSIONS, flat.size)
    if topk > 0:
        idxs = np.argpartition(-flat, topk - 1)[:topk]
        idxs = idxs[np.argsort(-flat[idxs])]
    else:
        idxs = np.array([], dtype=int)
    tcl = []
    print(f'\n=== Top-{topk} most commonly confused (True → Pred) across all seeds ===')
    for r, idx in enumerate(idxs, start=1):
        i = idx // num_classes
        j = idx % num_classes
        c = int(cm_off[i, j])
        if c <= 0:
            continue
        tcl.append((label_names[i], label_names[j], c))
        print(f'{r:2d}. {label_names[i]}  →  {label_names[j]}   (count={c})')
    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)
    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(y_true_all, y_pred_all, labels=np.arange(num_classes), zero_division=0)
    valid = per_support > 0
    f1_for_sort = per_f1.copy()
    f1_for_sort[~valid] = np.inf
    worst = np.argsort(f1_for_sort)[:max(1, REP_CLASSES // 3)]
    best = np.argsort(-per_f1)[:max(1, REP_CLASSES // 3)]
    mids = np.argsort(per_f1)
    mids = [i for i in mids if valid[i]]
    mid_take = mids[len(mids) // 2:len(mids) // 2 + max(1, REP_CLASSES // 3)]
    reps = []
    for arr in [worst, mid_take, best]:
        for i in arr:
            if valid[i] and i not in reps:
                reps.append(int(i))
    reps = reps[:REP_CLASSES]
    row_sums = cm_total.sum(axis=1)
    per_class_acc = np.divide(np.diag(cm_total), row_sums, out=np.zeros_like(row_sums, dtype=float), where=row_sums != 0)
    rdl = []
    print('\n=== Representative per-class performance (aggregated over seeds) ===')
    print('Disease | Support | Accuracy | Precision | Recall | F1')
    for i in reps:
        rdl.append((label_names[i], int(per_support[i]), float(per_class_acc[i]), float(per_p[i]), float(per_r[i]), float(per_f1[i])))
        print(f'{label_names[i]} | {int(per_support[i])} | {per_class_acc[i]:.4f} | {per_p[i]:.4f} | {per_r[i]:.4f} | {per_f1[i]:.4f}')
    out_prefix = os.path.join(OUTPUT_DIR, f'AEclf_full_block_confusion_g{GROUP_SIZE}_seeds{SEEDS[0]}-{SEEDS[-1]}')
    block_cm = aggregate_blocks(cm_total, GROUP_SIZE)
    plot_block_heatmap(block_cm, GROUP_SIZE, num_classes, NORMALIZE, out_prefix, test_metrics_summary=tms, confusion_stats=confusion_stats, top_confusions=tcl, rep_diseases=rdl)
    print(f'\nDone.')
    print(f'Saved heatmap: {out_prefix}.png and {out_prefix}.jpg')
    print(f'Total runtime: {time.time() - t0:.1f}s')
if __name__ == '__main__':
    main()
