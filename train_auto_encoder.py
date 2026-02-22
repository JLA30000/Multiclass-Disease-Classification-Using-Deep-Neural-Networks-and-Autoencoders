# train_autoencoder.py
# Trains an autoencoder with a fixed train/val/test split (from split_indices.npz),
# early stopping on VAL reconstruction loss, and final one-time test evaluation.

import argparse
import copy
import os
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

        # Safety for BCE (avoid log(0)); usually not needed but cheap + safe
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
        description="Train Autoencoder with fixed split + early stopping (VAL loss)")
    parser.add_argument("--data_path", type=str,
                        default="filtered_diseases.csv", help="CSV path")
    parser.add_argument("--indices_file", type=str,
                        default="split_indices.npz", help="NPZ with train/val/test indices")
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
    parser.add_argument("--seed", type=int, default=42,
                        help="Training seed (init + shuffle)")
    parser.add_argument("--out_dir", type=str,
                        default="runs_autoencoder", help="Output directory")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading datasets...")
    train_dataset = AutoencoderDataset(
        args.data_path, split="train", indices_file=args.indices_file)
    val_dataset = AutoencoderDataset(
        args.data_path, split="val",   indices_file=args.indices_file)
    test_dataset = AutoencoderDataset(
        args.data_path, split="test",  indices_file=args.indices_file)

    input_dim = train_dataset.X.shape[1]
    print(f"Input dimension: {input_dim}")

    # Seeded shuffling (important for multi-seed experiments)
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=g, num_workers=0)
    val_loader = DataLoader(
        val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = Autoencoder(input_dim=input_dim).to(device)

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
    print("\nInitial evaluation...")
    init_train_loss, init_train_bitacc = eval_loader(
        model, train_loader, criterion, device)
    init_val_loss, init_val_bitacc = eval_loader(
        model, val_loader,   criterion, device)

    print(
        f"Epoch 0 | Train Loss: {init_train_loss:.6f} | Val Loss: {init_val_loss:.6f}")
    print(
        f"        Train BitAcc: {init_train_bitacc:.4f} | Val BitAcc: {init_val_bitacc:.4f}")

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

        for batch_idx, data in enumerate(train_loader):
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

            if batch_idx % 50 == 0:
                print(
                    f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.6f}")

        avg_train_loss = running_loss / max(nb, 1)
        avg_train_bitacc = running_bitacc / max(nb, 1)

        avg_val_loss, avg_val_bitacc = eval_loader(
            model, val_loader, criterion, device)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_bitaccs.append(avg_train_bitacc)
        val_bitaccs.append(avg_val_bitacc)

        print(f"\nEpoch [{epoch}/{args.epochs}]")
        print(
            f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        print(
            f"Train BitAcc: {avg_train_bitacc:.4f} | Val BitAcc: {avg_val_bitacc:.4f}")
        print("-" * 60)

        # Early stopping on VAL loss (minimize)
        if epoch >= args.min_epochs:
            if (best_val_loss - avg_val_loss) > args.min_delta:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"  No VAL loss improvement: {patience_counter}/{args.patience}")
                if patience_counter >= args.patience:
                    print(f"  ✗ Early stopping at epoch {epoch}")
                    break

    # Restore best (by VAL loss)
    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"\nRestored best model from epoch {best_epoch} (best VAL loss: {best_val_loss:.6f})")
    else:
        print("\nWarning: best_state was never set (val loss never improved). Using last epoch weights.")

    # ---------- Final held-out test eval (only once) ----------
    test_loss, test_bitacc = eval_loader(model, test_loader, criterion, device)
    print("\n=== FINAL TEST (held-out) ===")
    print(f"Test Loss: {test_loss:.6f} | Test BitAcc: {test_bitacc:.4f}")

    # ---------- Save artifacts ----------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"seed{args.seed}_lr{args.lr}_bs{args.batch_size}_bestVAL{best_val_loss:.6f}"
    model_path = out_dir / f"autoencoder_{tag}_test{test_loss:.6f}.pth"
    plot_path = out_dir / f"autoencoder_{tag}.pdf"

    # torch.save(
    #     {
    #         "model_state_dict": model.state_dict(),
    #         "input_dim": input_dim,
    #         "seed": args.seed,
    #         "lr": args.lr,
    #         "batch_size": args.batch_size,
    #         "epochs_ran": len(train_losses) - 1,
    #         "best_epoch": best_epoch,
    #         "best_val_loss": best_val_loss,
    #         "final_test_loss": test_loss,
    #         "final_test_bitacc": test_bitacc,
    #         "train_losses": train_losses,
    #         "val_losses": val_losses,
    #         "train_bitaccs": train_bitaccs,
    #         "val_bitaccs": val_bitaccs,
    #         "indices_file": args.indices_file,
    #         "data_path": args.data_path,
    #     },
    #     model_path,
    # )

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
    # plt.savefig(plot_path, format="pdf", dpi=300, bbox_inches="tight")

    print("\nTraining completed!")
    print(f"Model saved to: {model_path}")
    print(f"Plot saved to:  {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
