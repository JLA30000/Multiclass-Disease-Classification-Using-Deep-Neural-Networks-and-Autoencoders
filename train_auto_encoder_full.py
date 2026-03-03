# train_autoencoder.py
# Trains an autoencoder with a fixed train/val/test split (from split_indices.npz),
# early stopping on VAL reconstruction loss, and final one-time test evaluation.

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from auto_encoder_dataset import AutoencoderDataset
from neural_network_autoencoder import Autoencoder


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bitwise_recon_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # outputs are probabilities in [0,1] because model ends with Sigmoid
    preds = (outputs >= 0.5).float()
    return (preds == targets).float().mean()


@torch.no_grad()
def eval_loader(model, loader, criterion, device, clamp_eps: float = 1e-7):
    model.eval()
    total_loss = 0.0
    total_bitacc = 0.0
    nb = 0

    for data in loader:
        data = data.to(device).float()
        outputs = model(data)

        # Safety for BCE (avoid log(0))
        outputs = outputs.clamp(clamp_eps, 1.0 - clamp_eps)

        loss = criterion(outputs, data)
        total_loss += loss.item()
        total_bitacc += bitwise_recon_accuracy(outputs, data).item()
        nb += 1

    if nb == 0:
        return float("inf"), 0.0

    return total_loss / nb, total_bitacc / nb


def main():
    parser = argparse.ArgumentParser(
        description="Train Autoencoder with fixed split + early stopping (VAL loss)"
    )
    parser.add_argument("--data_path", type=str,
                        default="diseases.csv", help="CSV path")
    parser.add_argument(
        "--indices_file",
        type=str,
        default="split_indices_full_80_10_10.npz",
        help="NPZ with train/val/test indices",
    )

    # ----- Hyperparams -----
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int,
                        default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--min_delta", type=float,
                        default=1e-4, help="Min VAL loss improvement")
    parser.add_argument("--min_epochs", type=int, default=10,
                        help="Minimum epochs before early stopping")

    # ----- NEW: latent size -----
    parser.add_argument("--latent_dim", type=int, default=64,
                        help="Bottleneck (latent) dimension")

    # ----- Reproducibility -----
    parser.add_argument("--seed", type=int, default=42,
                        help="Training seed (init + shuffle)")

    # ----- Output -----
    parser.add_argument("--out_dir", type=str,
                        default="runs_autoencoder", help="Output directory")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[START] seed={args.seed} lr={args.lr} batch_size={args.batch_size} "
        f"latent_dim={args.latent_dim} epochs={args.epochs} device={device}",
        flush=True,
    )
    train_dataset = AutoencoderDataset(
        args.data_path, split="train", indices_file=args.indices_file, verbose=False)
    val_dataset = AutoencoderDataset(
        args.data_path, split="val", indices_file=args.indices_file, verbose=False)
    test_dataset = AutoencoderDataset(
        args.data_path, split="test", indices_file=args.indices_file, verbose=False)

    input_dim = train_dataset.X.shape[1]

    # Seeded shuffling (important for multi-seed experiments)
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=g, num_workers=0)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(
        f"[DATA] train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)} "
        f"train_batches={len(train_loader)}",
        flush=True,
    )

    # ----- NEW: pass latent_dim into model -----
    model = Autoencoder(input_dim=input_dim,
                        latent_dim=args.latent_dim).to(device)

    # Because the model ends with Sigmoid and targets are binary, BCE is the natural choice.
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ---------- Trackers ----------
    train_losses, val_losses = [], []
    train_bitaccs, val_bitaccs = [], []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state = None

    # ---------- Initial eval ----------
    init_train_loss, init_train_bitacc = eval_loader(
        model, train_loader, criterion, device)
    init_val_loss, init_val_bitacc = eval_loader(
        model, val_loader, criterion, device)
    print(
        "[INIT] "
        f"train_loss={init_train_loss:.6f} train_bitacc={init_train_bitacc:.4f} "
        f"val_loss={init_val_loss:.6f} val_bitacc={init_val_bitacc:.4f}",
        flush=True,
    )

    train_losses.append(init_train_loss)
    val_losses.append(init_val_loss)
    train_bitaccs.append(init_train_bitacc)
    val_bitaccs.append(init_val_bitacc)

    # ---------- Train loop ----------
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_bitacc = 0.0
        nb = 0

        for data in train_loader:
            data = data.to(device).float()
            optimizer.zero_grad()

            outputs = model(data).clamp(1e-7, 1.0 - 1e-7)
            loss = criterion(outputs, data)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_bitacc += bitwise_recon_accuracy(
                outputs.detach(), data).item()
            nb += 1

        avg_train_loss = running_loss / max(nb, 1)
        avg_train_bitacc = running_bitacc / max(nb, 1)

        avg_val_loss, avg_val_bitacc = eval_loader(
            model, val_loader, criterion, device)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_bitaccs.append(avg_train_bitacc)
        val_bitaccs.append(avg_val_bitacc)

        # Always track the best model state
        stop_now = False
        if (best_val_loss - avg_val_loss) > args.min_delta:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            # Only count patience after min_epochs
            if epoch >= args.min_epochs:
                patience_counter += 1
                if patience_counter >= args.patience:
                    stop_now = True
        if stop_now:
            break

    # Restore best (by VAL loss)
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = int(np.argmin(val_losses))
        best_val_loss = float(val_losses[best_epoch])

    # ---------- Final held-out test eval (only once) ----------
    test_loss, test_bitacc = eval_loader(model, test_loader, criterion, device)
    print(
        f"[TEST] loss={test_loss:.6f} bitacc={test_bitacc:.4f}",
        flush=True,
    )

    # ---------- Save artifacts ----------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- NEW: include z (latent_dim) in tag -----
    tag = (
        f"seed{args.seed}_lr{args.lr}_bs{args.batch_size}_z{args.latent_dim}"
        f"_bestVAL{best_val_loss:.6f}"
    )
    model_path = out_dir / f"autoencoder_{tag}_test{test_loss:.6f}.pth"
    plot_path = out_dir / f"autoencoder_{tag}.pdf"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "latent_dim": args.latent_dim,  # NEW
            "seed": args.seed,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs_ran": len(train_losses) - 1,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "final_test_loss": test_loss,
            "final_test_bitacc": test_bitacc,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_bitaccs": train_bitaccs,
            "val_bitaccs": val_bitaccs,
            "indices_file": args.indices_file,
            "data_path": args.data_path,
        },
        model_path,
    )

    # ---------- Plots ----------
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Val Loss", linewidth=2)
    if best_epoch > 0:
        plt.axvline(x=best_epoch, linestyle="--", alpha=0.5,
                    label=f"Best Epoch {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("Autoencoder Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(train_bitaccs, label="Train BitAcc", linewidth=2)
    plt.plot(val_bitaccs, label="Val BitAcc", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Bitwise Reconstruction Accuracy")
    plt.title("Reconstruction Accuracy (threshold=0.5)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"[SAVED] model={model_path}", flush=True)
    print(f"[SAVED] plot={plot_path}", flush=True)

    print(
        "BEST_VAL_LOSS "
        f"lr={args.lr} latent_dim={args.latent_dim} batch_size={args.batch_size} "
        f"seed={args.seed} best_val_loss={best_val_loss:.6f}"
    )


if __name__ == "__main__":
    main()
