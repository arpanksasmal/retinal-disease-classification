"""
train.py - Training Entry Point v4
====================================
FIXES vs v3:
  1. WeightedRandomSampler REMOVED  - was causing model collapse (predicting 1 class)
  2. Class weights in loss ONLY     - single correction for imbalance (correct way)
  3. All EfficientNet layers unfrozen - more capacity to learn
  4. NUM_WORKERS = 2                - stable on Windows Anaconda
  5. LABEL_SMOOTHING = 0.05         - reduced for stability
  6. GPU VRAM printed each epoch    - monitor usage

Run:
    python train.py

Requires:
    - data/train.csv
    - data/train_images/

Outputs:
    - saved_models/best_model.pt
    - plots/training_history.png
    - plots/confusion_matrix.png
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config  import Config
from model   import RetinalCNN
from dataset import RetinalDataset, get_train_transforms, get_val_transforms
from utils   import (
    set_seed,
    EarlyStopping,
    plot_training,
    plot_confusion,
    print_report,
)


# --- GPU Monitor -------------------------------------------------------------

def print_gpu_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved  = torch.cuda.memory_reserved()  / 1024**3
        total     = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU VRAM: {allocated:.2f}GB allocated | "
              f"{reserved:.2f}GB reserved | {total:.2f}GB total")


# --- One Epoch Helpers -------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    all_preds  = []
    all_labels = []

    for images, labels in tqdm(loader, desc="  val  ", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# --- Main --------------------------------------------------------------------

def main():
    cfg = Config()
    set_seed(cfg.SEED)

    print(f"\n{'='*60}")
    print(f"  Retinal Disease Detection --- Training v4")
    print(f"  Device     : {cfg.DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  Total VRAM : {total_vram:.1f} GB")
    print(f"  Batch size : {cfg.BATCH_SIZE}")
    print(f"  Workers    : {cfg.NUM_WORKERS}")
    print(f"  LR         : {cfg.LEARNING_RATE}")
    print(f"  Sampler    : Shuffle (WeightedSampler OFF - class weights in loss)")
    print(f"  Patience   : {cfg.PATIENCE}")
    print(f"{'='*60}\n")

    # Data
    if not os.path.exists(cfg.TRAIN_CSV):
        sys.exit(
            f"[ERROR] CSV not found at '{cfg.TRAIN_CSV}'.\n"
            "Download APTOS 2019 from Kaggle."
        )

    df = pd.read_csv(cfg.TRAIN_CSV)
    print(f"Total samples  : {len(df)}")
    print(f"Class counts   :\n{df['diagnosis'].value_counts().sort_index().to_string()}\n")

    # Class weights for loss - SINGLE correction for imbalance
    counts        = df["diagnosis"].value_counts().sort_index().values.astype(float)
    class_weights = torch.FloatTensor(
        counts.sum() / (len(counts) * counts)
    ).to(cfg.DEVICE)
    print(f"Class weights  : {np.round(class_weights.cpu().numpy(), 3)}")
    print(f"NOTE: Using class weights in loss only (no sampler) - prevents collapse\n")

    train_df, val_df = train_test_split(
        df,
        test_size    = cfg.VALIDATION_SPLIT,
        stratify     = df["diagnosis"],
        random_state = cfg.SEED,
    )
    print(f"Train : {len(train_df)} | Val : {len(val_df)}\n")

    # Datasets
    train_ds = RetinalDataset(
        train_df, cfg.TRAIN_IMG_DIR, get_train_transforms(cfg.IMG_SIZE)
    )
    val_ds = RetinalDataset(
        val_df, cfg.TRAIN_IMG_DIR, get_val_transforms(cfg.IMG_SIZE)
    )

    # Loaders - simple shuffle, no sampler
    train_loader = DataLoader(
        train_ds,
        batch_size         = cfg.BATCH_SIZE,
        shuffle            = True,
        num_workers        = cfg.NUM_WORKERS,
        pin_memory         = cfg.PIN_MEMORY,
        persistent_workers = cfg.NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size         = cfg.BATCH_SIZE,
        shuffle            = False,
        num_workers        = cfg.NUM_WORKERS,
        pin_memory         = cfg.PIN_MEMORY,
        persistent_workers = cfg.NUM_WORKERS > 0,
    )

    # Model
    model     = RetinalCNN(num_classes=cfg.NUM_CLASSES).to(cfg.DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"Trainable params : {trainable:,} / {total_p:,} total")
    print_gpu_stats()
    print()

    # Loss - weighted CrossEntropy handles imbalance
    criterion = nn.CrossEntropyLoss(
        weight          = class_weights,
        label_smoothing = cfg.LABEL_SMOOTHING,
    )

    # Optimizer - AdamW over all trainable params
    optimizer = optim.AdamW(
        model.parameters(),
        lr           = cfg.LEARNING_RATE,
        weight_decay = cfg.WEIGHT_DECAY,
    )

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        factor   = cfg.SCHEDULER_FACTOR,
        patience = cfg.SCHEDULER_PATIENCE,
        min_lr   = cfg.SCHEDULER_MIN_LR,
        verbose  = True,
    )

    # Early Stopping
    early_stop = EarlyStopping(
        patience  = cfg.PATIENCE,
        min_delta = cfg.MIN_DELTA,
        save_path = cfg.MODEL_SAVE_PATH,
    )

    # Training Loop
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    print("Starting training ...\n")

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:>3}/{cfg.NUM_EPOCHS}  |  LR = {current_lr:.2e}")

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, cfg.DEVICE
        )
        vl_loss, vl_acc, vl_preds, vl_labels = validate(
            model, val_loader, criterion, cfg.DEVICE
        )

        scheduler.step(vl_loss)

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);   val_accs.append(vl_acc)

        print(
            f"  Train --- Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f}\n"
            f"  Val   --- Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}"
        )
        print_gpu_stats()

        early_stop(vl_loss, model)
        if early_stop.early_stop:
            print(f"\n Early stopping triggered at epoch {epoch}.")
            break

        print()

    # Final Evaluation
    print("\n" + "="*60)
    print("  Loading best model for final evaluation ...")
    model.load_state_dict(
        torch.load(cfg.MODEL_SAVE_PATH, map_location=cfg.DEVICE)
    )
    _, best_acc, best_preds, best_labels = validate(
        model, val_loader, criterion, cfg.DEVICE
    )
    print(f"  Best Val Accuracy : {best_acc:.4f}")

    print_report(best_labels, best_preds, cfg.CLASS_NAMES)
    plot_training(train_losses, val_losses, train_accs, val_accs, cfg.PLOTS_DIR)
    plot_confusion(best_labels, best_preds, cfg.CLASS_NAMES, cfg.PLOTS_DIR)

    print("\n Training complete. Check 'saved_models/' and 'plots/' for outputs.")


# CRITICAL for Windows: must be inside __main__ guard
if __name__ == "__main__":
    main()
