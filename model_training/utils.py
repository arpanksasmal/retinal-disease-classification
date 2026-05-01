"""
Utility helpers:
  - set_seed          : reproducibility
  - EarlyStopping     : monitors val_loss, saves best checkpoint
  - plot_training     : loss & accuracy curves
  - plot_confusion    : confusion matrix heatmap
  - print_report      : sklearn classification report
"""

import os
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Early Stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when validation loss stops improving.

    Parameters
    ----------
    patience   : epochs to wait without improvement before stopping
    min_delta  : minimum change to count as an improvement
    save_path  : where to persist the best model weights
    """

    def __init__(
        self,
        patience: int  = 12,
        min_delta: float = 1e-4,
        save_path: str  = "saved_models/best_model.pt",
    ):
        self.patience   = patience
        self.min_delta  = min_delta
        self.save_path  = save_path
        self.counter    = 0
        self.best_loss  = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found → save & reset counter
            self.best_loss = val_loss
            self._save(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f"  EarlyStopping: no improvement for {self.counter}/{self.patience} epoch(s). "
                f"Best val_loss = {self.best_loss:.5f}"
            )
            if self.counter >= self.patience:
                self.early_stop = True

    def _save(self, model: torch.nn.Module) -> None:
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(model.state_dict(), self.save_path)
        print(f"  ✓ Best model saved → {self.save_path}  (val_loss={self.best_loss:.5f})")


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_training(
    train_losses, val_losses,
    train_accs,   val_accs,
    save_dir: str = "plots",
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(train_losses, label="Train Loss", color="#2196F3", linewidth=2)
    axes[0].plot(val_losses,   label="Val Loss",   color="#F44336", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(train_accs, label="Train Acc", color="#2196F3", linewidth=2)
    axes[1].plot(val_accs,   label="Val Acc",   color="#F44336", linewidth=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, "training_history.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved → {out}")


def plot_confusion(
    y_true, y_pred,
    class_names,
    save_dir: str = "plots",
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    out = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved → {out}")


# ─── Metrics ──────────────────────────────────────────────────────────────────

def print_report(y_true, y_pred, class_names) -> None:
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names))
