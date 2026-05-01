# 👁️ Retinal Disease Detection — Diabetic Retinopathy Grading

> **Custom CNN with CBAM Attention** for automated grading of Diabetic Retinopathy (DR) severity from retinal fundus photographs.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)

---

## 📌 Project Overview

Diabetic Retinopathy is a leading cause of blindness worldwide. Early automated detection can significantly reduce vision loss. This project builds an end-to-end deep learning pipeline that:

- Classifies retinal fundus images into **5 DR severity grades** (0–4)
- Uses a **custom CNN backbone** with **CBAM (Convolutional Block Attention Module)** for spatial and channel-wise attention
- Implements **early stopping, class-weighted loss, and cosine LR scheduling** to prevent overfitting
- Provides a **production-ready Streamlit web app** for real-time inference

**Architecture highlight:** CBAM forces the model to focus on clinically relevant regions (microaneurysms, haemorrhages, neovascularisation) rather than background noise.

---

## 📁 Folder Structure

```
retinal-disease-detection/
│
├── model_training/                 # Training pipeline
│   ├── config.py                   # All hyperparameters in one place
│   ├── dataset.py                  # PyTorch Dataset + augmentations
│   ├── model.py                    # RetinalCNN architecture with CBAM
│   ├── utils.py                    # EarlyStopping, plots, metrics
│   ├── train.py                    # Main training entry point
│   ├── saved_models/               # best_model.pt saved here after training
│   ├── plots/                      # training_history.png, confusion_matrix.png
│   └── requirements.txt
│
├── streamlit_app/                  # Inference web application
│   ├── app.py                      # Streamlit UI
│   ├── model.py                    # Model definition (self-contained copy)
│   └── requirements.txt
│
├── requirements.txt                # Combined requirements
└── README.md
```

---

## 📦 Dataset

### Download (Free, Kaggle)

**APTOS 2019 Blindness Detection Dataset**
🔗 https://www.kaggle.com/competitions/aptos2019-blindness-detection/data

You need a free Kaggle account to download.

### Dataset Details

| Property | Value |
|---|---|
| Total images | 3,662 |
| Image format | PNG (retinal fundus) |
| Task | 5-class classification |
| Labels | 0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative |

### After Downloading

Extract and place files like this:

```
retinal-disease-detection/
└── model_training/
    └── data/
        ├── train.csv           ← from Kaggle zip
        └── train_images/       ← from Kaggle zip
            ├── 000c1434d8d7.png
            ├── 001639a390f0.png
            └── ... (3662 images)
```

> `train.csv` has two columns: `id_code` (image filename without extension) and `diagnosis` (0–4 label).

---

## 🏗️ Model Architecture

```
Input (3 × 224 × 224)
        │
    ┌───▼───────────────────────────────────────────┐
    │  Stem: Conv7x7 (stride 2) → BN → ReLU →       │
    │         MaxPool (stride 2)                     │  → 32ch, 56×56
    └───────────────────────────────────────────────┘
        │
    ┌───▼───────────┐  ┌──────┐
    │  ResidualCBAM │→ │ CBAM │  Layer 1: 64ch,  56×56
    └───────────────┘  └──────┘
        │
    ┌───▼───────────┐  ┌──────┐
    │  ResidualCBAM │→ │ CBAM │  Layer 2: 128ch, 28×28  (stride-2)
    └───────────────┘  └──────┘
        │
    ┌───▼───────────┐  ┌──────┐
    │  ResidualCBAM │→ │ CBAM │  Layer 3: 256ch, 14×14  (stride-2)
    └───────────────┘  └──────┘
        │
    ┌───▼───────────┐  ┌──────┐
    │  ResidualCBAM │→ │ CBAM │  Layer 4: 512ch, 7×7   (stride-2)
    └───────────────┘  └──────┘
        │
    Global AvgPool → Dropout(0.5)
        │
    FC(512→256) → ReLU → Dropout(0.3)
        │
    FC(256→5) → Logits
```

### CBAM (Convolutional Block Attention Module)

CBAM applies two sequential attention mechanisms:
1. **Channel Attention** — "which feature maps are important?" (shared MLP on avg + max pooled features)
2. **Spatial Attention** — "where in the image to focus?" (conv on channel-wise pooled maps)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download & Place Dataset

Follow the **Dataset** section above.

### 3. Train the Model

```bash
cd model_training
python train.py
```

Training will:
- Print per-epoch train/val loss and accuracy
- Save the **best checkpoint** to `saved_models/best_model.pt` automatically
- Stop early if validation loss stops improving (patience = 12 epochs)
- Save `plots/training_history.png` and `plots/confusion_matrix.png`

### 4. Run Streamlit App

```bash
cd streamlit_app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

- The app auto-loads the model from `../model_training/saved_models/best_model.pt`
- Or you can **upload your own `.pth` file** via the sidebar

---

## ⚙️ Configuration

All hyperparameters live in `model_training/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `IMG_SIZE` | 224 | Input image resolution |
| `BATCH_SIZE` | 32 | Training batch size |
| `NUM_EPOCHS` | 60 | Maximum epochs |
| `LEARNING_RATE` | 1e-3 | Initial AdamW LR |
| `WEIGHT_DECAY` | 1e-4 | L2 regularisation |
| `PATIENCE` | 12 | Early stopping patience |
| `MIN_DELTA` | 1e-4 | Min improvement to reset counter |
| `VALIDATION_SPLIT` | 0.20 | 80/20 train-val split |
| `NUM_WORKERS` | 0 | DataLoader workers (set to 4 on Linux) |

---

## 🛡️ Overfitting Prevention Strategy

| Technique | Where |
|---|---|
| **Early Stopping** | Stops if val_loss doesn't improve for 12 epochs |
| **CosineAnnealingLR** | Smoothly decays LR to prevent late overfitting |
| **Dropout (0.5 + 0.3)** | Applied in classifier head |
| **Data Augmentation** | Flips, rotations, colour jitter, affine transforms |
| **Class-weighted CrossEntropy** | Handles class imbalance |
| **Gradient Clipping** | `max_norm=1.0` to prevent exploding gradients |

---

## 📊 Expected Results

| Metric | Expected Range |
|---|---|
| Val Accuracy | 75–85% |
| Best Epoch | 20–40 (with early stopping) |
| Training time (CPU) | ~3–5 hrs |
| Training time (GPU) | ~20–40 mins |

> Results depend on hardware and random seed. GPU strongly recommended.

---

## 🔬 Tech Stack

- **PyTorch** — model training
- **Torchvision** — transforms
- **Scikit-learn** — train/val split, metrics
- **Seaborn / Matplotlib** — confusion matrix, training curves
- **Streamlit** — inference web app
- **Pandas / NumPy** — data handling

---

## ⚠️ Disclaimer

This project is for **research and educational purposes only**.
It is **not** a clinical diagnostic tool. Do not use model outputs to make medical decisions.
