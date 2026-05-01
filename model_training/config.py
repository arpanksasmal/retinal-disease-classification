"""
Config v4 - Fixed collapse issues
KEY FIXES:
  - USE_WEIGHTED_SAMPLER = False  (was causing model collapse with class weights)
  - Class weights in loss ONLY    (single correction, not double)
  - NUM_WORKERS = 2               (4 causes issues on Windows/Anaconda)
  - LR = 2e-4                     (slightly higher, all layers now trainable)
  - PATIENCE = 12                 (more room to breathe)
  - BATCH_SIZE = 32               (safer for stability)
"""

import os
import torch


class Config:
    # Paths
    DATA_DIR        = "data"
    TRAIN_CSV       = os.path.join("data", "train.csv")
    TRAIN_IMG_DIR   = os.path.join("data", "train_images")
    MODEL_SAVE_PATH = os.path.join("saved_models", "best_model.pt")
    PLOTS_DIR       = "plots"

    # Dataset
    IMG_SIZE         = 224
    NUM_CLASSES      = 5
    CLASS_NAMES      = [
        "No DR",
        "Mild DR",
        "Moderate DR",
        "Severe DR",
        "Proliferative DR",
    ]
    VALIDATION_SPLIT = 0.20

    # Training
    BATCH_SIZE     = 32           # stable for fine-tuning
    NUM_EPOCHS     = 50
    LEARNING_RATE  = 2e-4         # slightly higher since all layers trainable
    WEIGHT_DECAY   = 1e-4
    NUM_WORKERS    = 2            # 4 causes issues on Windows Anaconda
    PIN_MEMORY     = True

    # Early Stopping
    PATIENCE       = 12
    MIN_DELTA      = 1e-4

    # Scheduler
    SCHEDULER_FACTOR    = 0.5
    SCHEDULER_PATIENCE  = 4
    SCHEDULER_MIN_LR    = 1e-7

    # Loss
    LABEL_SMOOTHING = 0.05        # reduced - less smoothing more stable

    # MixUp - OFF
    USE_MIXUP       = False

    # Weighted Sampler - OFF
    # CRITICAL FIX: sampler + class weights in loss = double correction = collapse
    # Using class weights in loss ONLY is the correct single correction
    USE_WEIGHTED_SAMPLER = False

    # Reproducibility
    SEED = 42

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
