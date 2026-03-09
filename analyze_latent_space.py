import argparse
import glob
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, precision_recall_fscore_support
from scipy.stats import spearmanr
from dataset import diagnosticsDataset
from neural_network_autoencoder import Autoencoder

AE_CKPT_GLOB_DEFAULT = 'runs_autoencoder/autoencoder_seed0_lr0.0001_bs64_z64_bestVAL*.pth'
AE_CKPT_LEGACY_FALLBACK = 'runs_autoencoder/autoencoder_seed0_lr0.0001_bs64_z128_bestVAL0.000248_test0.000235.pth'


def resolve_ae_checkpoint(requested_path: str) -> str:
    raw = (requested_path or '').strip()
    if not raw or raw.lower() == 'auto':
        candidates = sorted(glob.glob(AE_CKPT_GLOB_DEFAULT))
        if candidates:
            return candidates[0]
        if os.path.isfile(AE_CKPT_LEGACY_FALLBACK):
            return AE_CKPT_LEGACY_FALLBACK
        raise FileNotFoundError(f"No checkpoint found via '{AE_CKPT_GLOB_DEFAULT}' or fallback.")
    if any((ch in raw for ch in '*?[]')):
        candidates = sorted(glob.glob(raw))
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f'No checkpoint matched: {raw}')
    if os.path.isfile(raw):
        return raw
    raise FileNotFoundError(f'Checkpoint not found: {raw}')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()


def extract_features(ae: nn.Module, dataset, device: torch.device, batch_size: int=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    raw_list, latent_list, label_list = ([], [], [])
    ae.eval()
    for X, y in loader:
        X = X.to(device)
        z = ae.encoder(X)
        raw_list.append(X.cpu().numpy())
        latent_list.append(z.cpu().numpy())
        label_list.append(y.numpy())
    return (np.concatenate(raw_list), np.concatenate(latent_list), np.concatenate(label_list))

@torch.no_grad()


def per_sample_reconstruction_error(ae: nn.Module, dataset, device: torch.device, batch_size: int=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    errors = []
    ae.eval()
    for X, y in loader:
        X = X.to(device)
        recon = ae(X)
        mse = ((X - recon) ** 2).mean(dim=1)
        errors.append(mse.cpu().numpy())
    return np.concatenate(errors)


def analysis_pca(raw: np.ndarray, latent: np.ndarray, labels: np.ndarray, label_names, top_n_classes: int, out_dir: str):
    print('\n' + '=' * 60)
    print('Analysis 1: PCA Comparison (raw vs. latent)')
    print('=' * 60)
    pca_raw = PCA(n_components=min(50, raw.shape[1]))
    pca_lat = PCA(n_components=min(50, latent.shape[1]))
    raw_2d = pca_raw.fit_transform(raw)[:, :2]
    lat_2d = pca_lat.fit_transform(latent)[:, :2]
    var_raw_2 = pca_raw.explained_variance_ratio_[:2].sum()
    var_lat_2 = pca_lat.explained_variance_ratio_[:2].sum()
    var_raw_10 = pca_raw.explained_variance_ratio_[:10].sum()
    var_lat_10 = pca_lat.explained_variance_ratio_[:min(10, latent.shape[1])].sum()
    print(f'Variance explained (2 PCs)  — raw: {var_raw_2:.4f}  latent: {var_lat_2:.4f}')
    print(f'Variance explained (10 PCs) — raw: {var_raw_10:.4f}  latent: {var_lat_10:.4f}')
    sil_n = min(8000, len(labels))
    rng = np.random.RandomState(42)
    sil_idx = rng.choice(len(labels), sil_n, replace=False)
    sil_raw = silhouette_score(raw_2d[sil_idx], labels[sil_idx])
    sil_lat = silhouette_score(lat_2d[sil_idx], labels[sil_idx])
    print(f'Silhouette score (2-D PCA)  — raw: {sil_raw:.4f}  latent: {sil_lat:.4f}')
    class_counts = np.bincount(labels, minlength=len(label_names))
    top_cls = np.argsort(-class_counts)[:top_n_classes]
    mask = np.isin(labels, top_cls)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    cmap = plt.cm.get_cmap('tab20', top_n_classes)
    cls_to_idx = {c: i for i, c in enumerate(top_cls)}
    colors = np.array([cls_to_idx[l] for l in labels[mask]])
    for ax, pts, title in [(axes[0], raw_2d[mask], 'Raw Features — PCA 2-D'), (axes[1], lat_2d[mask], 'Latent Features — PCA 2-D')]:
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=colors, cmap=cmap, s=6, alpha=0.5, edgecolors='none')
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
    handles = []
    for i, c in enumerate(top_cls):
        name = label_names[c]
        short = name[:22] + '..' if len(name) > 24 else name
        handles.append(plt.Line2D([], [], marker='o', linestyle='None', color=cmap(i), label=short, markersize=5))
    fig.legend(handles=handles, loc='lower center', ncol=min(5, top_n_classes), fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f'PCA Comparison — Silhouette: raw {sil_raw:.3f} → latent {sil_lat:.3f}  |  Var. explained (2 PCs): raw {var_raw_2:.1%} → latent {var_lat_2:.1%}', fontsize=11, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, 'pca_comparison')
    plt.savefig(path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(path + '.jpg', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}.png / .jpg')
    return {'sil_raw': sil_raw, 'sil_latent': sil_lat, 'var_raw_2pc': var_raw_2, 'var_lat_2pc': var_lat_2, 'var_raw_10pc': var_raw_10, 'var_lat_10pc': var_lat_10}


def analysis_tsne_umap(raw: np.ndarray, latent: np.ndarray, labels: np.ndarray, label_names, top_n_classes: int, out_dir: str, use_umap: bool=False, perplexity: int=30):
    method = 'UMAP' if use_umap else 't-SNE'
    print(f"\n{'=' * 60}")
    print(f'Analysis 2: {method} Visualization (raw vs. latent)')
    print('=' * 60)
    class_counts = np.bincount(labels, minlength=len(label_names))
    top_cls = np.argsort(-class_counts)[:top_n_classes]
    mask = np.isin(labels, top_cls)
    raw_sub = raw[mask]
    lat_sub = latent[mask]
    lab_sub = labels[mask]
    max_pts = 10000
    if len(lab_sub) > max_pts:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(lab_sub), max_pts, replace=False)
        raw_sub = raw_sub[idx]
        lat_sub = lat_sub[idx]
        lab_sub = lab_sub[idx]
    print(f'Using {len(lab_sub)} points from top-{top_n_classes} classes')
    if use_umap:
        try:
            import umap
        except ImportError:
            print('WARNING: umap-learn not installed, falling back to t-SNE')
            use_umap = False
            method = 't-SNE'
    if use_umap:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        raw_2d = reducer.fit_transform(raw_sub)
        lat_2d = reducer.fit_transform(lat_sub)
    else:
        raw_2d = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto').fit_transform(raw_sub)
        lat_2d = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto').fit_transform(lat_sub)
    cmap = plt.cm.get_cmap('tab20', top_n_classes)
    cls_to_idx = {c: i for i, c in enumerate(top_cls)}
    colors = np.array([cls_to_idx[l] for l in lab_sub])
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, pts, title in [(axes[0], raw_2d, f'Raw Features — {method}'), (axes[1], lat_2d, f'Latent Features — {method}')]:
        ax.scatter(pts[:, 0], pts[:, 1], c=colors, cmap=cmap, s=8, alpha=0.55, edgecolors='none')
        ax.set_title(title, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
    handles = []
    for i, c in enumerate(top_cls):
        name = label_names[c]
        short = name[:22] + '..' if len(name) > 24 else name
        handles.append(plt.Line2D([], [], marker='o', linestyle='None', color=cmap(i), label=short, markersize=5))
    fig.legend(handles=handles, loc='lower center', ncol=min(5, top_n_classes), fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f'{method} — Top {top_n_classes} Most Frequent Classes', fontsize=12, y=1.01)
    plt.tight_layout()
    tag = method.lower().replace('-', '')
    path = os.path.join(out_dir, f'{tag}_comparison')
    plt.savefig(path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(path + '.jpg', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}.png / .jpg')


def analysis_recon_vs_f1(ae: nn.Module, test_ds, labels: np.ndarray, label_names, device: torch.device, out_dir: str, min_support: int=5):
    print('\n' + '=' * 60)
    print('Analysis 3: Reconstruction Error vs. Classification F1')
    print('=' * 60)
    num_classes = len(label_names)
    recon_err = per_sample_reconstruction_error(ae, test_ds, device)
    per_class_recon = np.full(num_classes, np.nan)
    pcs = np.zeros(num_classes, dtype=int)
    for c in range(num_classes):
        mask = labels == c
        pcs[c] = mask.sum()
        if mask.sum() > 0:
            per_class_recon[c] = recon_err[mask].mean()
    print('Computing per-class F1 via 1-NN in latent space on the test set ...')
    per_class_f1 = _compute_f1_via_knn(ae, test_ds, device, num_classes)
    valid = (pcs >= min_support) & ~np.isnan(per_class_recon) & ~np.isnan(per_class_f1)
    rc = per_class_recon[valid]
    f1 = per_class_f1[valid]
    names_valid = [label_names[i] for i in range(num_classes) if valid[i]]
    rho, pval = spearmanr(rc, f1)
    print(f'Classes used: {valid.sum()} / {num_classes}  (min support={min_support})')
    print(f'Spearman rho = {rho:.4f}   p-value = {pval:.2e}')
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(rc, f1, s=20, alpha=0.6, edgecolors='k', linewidths=0.3)
    if len(rc) > 2:
        z = np.polyfit(rc, f1, 1)
        xs = np.linspace(rc.min(), rc.max(), 100)
        ax.plot(xs, np.polyval(z, xs), 'r--', linewidth=1.5, label=f'Linear fit (Spearman ρ = {rho:.3f}, p = {pval:.2e})')
    n_annotate = 5
    worst_recon = np.argsort(-rc)[:n_annotate]
    best_f1 = np.argsort(-f1)[:n_annotate]
    worst_f1 = np.argsort(f1)[:n_annotate]
    annotate_idx = set(list(worst_recon) + list(best_f1) + list(worst_f1))
    for i in annotate_idx:
        name = names_valid[i]
        short = name[:20] + '..' if len(name) > 22 else name
        ax.annotate(short, (rc[i], f1[i]), fontsize=5.5, alpha=0.75, xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel('Per-Class Mean Reconstruction Error (MSE)', fontsize=11)
    ax.set_ylabel('Per-Class F1 Score', fontsize=11)
    ax.set_title(f'Reconstruction Quality vs. Classification Performance\nSpearman ρ = {rho:.3f}, p = {pval:.2e}  ({valid.sum()} classes)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'recon_vs_f1')
    plt.savefig(path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(path + '.jpg', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}.png / .jpg')
    return {'spearman_rho': rho, 'spearman_pval': pval}


def _compute_f1_via_knn(ae: nn.Module, test_ds, device: torch.device, num_classes: int):
    from sklearn.neighbors import KNeighborsClassifier
    loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    z_list, y_list = ([], [])
    ae.eval()
    with torch.no_grad():
        for X, y in loader:
            z_list.append(ae.encoder(X.to(device)).cpu().numpy())
            y_list.append(y.numpy())
    Z = np.concatenate(z_list)
    Y = np.concatenate(y_list)
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(Y))
    split = int(0.8 * len(Y))
    Z_fit, Y_fit = (Z[idx[:split]], Y[idx[:split]])
    Z_eval, Y_eval = (Z[idx[split:]], Y[idx[split:]])
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean', n_jobs=-1)
    knn.fit(Z_fit, Y_fit)
    preds = knn.predict(Z_eval)
    _, _, f1, _ = precision_recall_fscore_support(Y_eval, preds, labels=np.arange(num_classes), zero_division=0)
    return f1


def write_summary(pca_results: dict, corr_results: dict, out_dir: str):
    path = os.path.join(out_dir, 'latent_analysis_summary.txt')
    lines = ['Latent Space Quality — Summary', '=' * 50, '', '1. PCA Comparison', f"   Silhouette score (2-D PCA) — raw: {pca_results['sil_raw']:.4f}  latent: {pca_results['sil_latent']:.4f}", f"   Variance explained (2 PCs) — raw: {pca_results['var_raw_2pc']:.4f}  latent: {pca_results['var_lat_2pc']:.4f}", f"   Variance explained (10 PCs) — raw: {pca_results['var_raw_10pc']:.4f}  latent: {pca_results['var_lat_10pc']:.4f}", '', '2. t-SNE / UMAP', '   See saved figure for visual comparison.', '', '3. Reconstruction Error vs. Classification F1', f"   Spearman rho = {corr_results['spearman_rho']:.4f}  (p = {corr_results['spearman_pval']:.2e})", '   Negative rho → classes the AE reconstructs well tend to classify better.', '', '=' * 50]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\nSummary written to {path}')


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    parser = argparse.ArgumentParser(description='Latent space quality analysis for the autoencoder')
    parser.add_argument('--data_path', type=str, default='diseases.csv')
    parser.add_argument('--indices_file', type=str, default='split_indices_full_80_10_10.npz')
    parser.add_argument('--ae_checkpoint', type=str, default='auto')
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--top_classes', type=int, default=15, help='Number of most-frequent classes to show in scatter plots')
    parser.add_argument('--use_umap', action='store_true', help='Use UMAP instead of t-SNE for Analysis 2')
    parser.add_argument('--perplexity', type=int, default=30, help='t-SNE perplexity (ignored if --use_umap)')
    parser.add_argument('--min_support', type=int, default=5, help='Min test-set samples for a class to be included in Analysis 3')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default='runs_latent_analysis')
    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    test_ds = diagnosticsDataset(args.data_path, split='test', indices_file=args.indices_file)
    label_names = list(test_ds.label_names)
    num_classes = len(label_names)
    input_dim = test_ds.features.shape[1]
    print(f'Test set: {len(test_ds)} samples, {input_dim} features, {num_classes} classes')
    ckpt_path = resolve_ae_checkpoint(args.ae_checkpoint)
    print(f'Loading autoencoder: {ckpt_path}')
    ae = Autoencoder(input_dim=input_dim, latent_dim=args.latent_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    ae.load_state_dict(state)
    ae.eval()
    print(f'Autoencoder loaded (latent_dim={args.latent_dim})')
    print('\nExtracting features ...')
    raw, latent, labels = extract_features(ae, test_ds, device)
    print(f'Raw shape: {raw.shape}  Latent shape: {latent.shape}')
    pca_results = analysis_pca(raw, latent, labels, label_names, args.top_classes, args.out_dir)
    analysis_tsne_umap(raw, latent, labels, label_names, args.top_classes, args.out_dir, use_umap=args.use_umap, perplexity=args.perplexity)
    corr_results = analysis_recon_vs_f1(ae, test_ds, labels, label_names, device, args.out_dir, min_support=args.min_support)
    write_summary(pca_results, corr_results, args.out_dir)
    print('\nAll analyses complete.')
if __name__ == '__main__':
    main()
