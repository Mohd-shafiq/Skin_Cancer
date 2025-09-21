# Skin Cancer Detection 

A PyTorch project for classifying dermatoscopic images from the HAM10000 dataset into 7 lesion types using a ResNet50 backbone, with a simple Gradio demo for inference.

- Classes: `['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']`
- Considered cancerous (for quick user guidance): `{'akiec', 'bcc', 'mel'}`

> Note: This repository contains a training notebook and a small deployment script that uses a saved model file `skin_cancer_model.pth`. Results will vary by training settings and hardware; the notebook reports a high accuracy in the experiments shown but your results may differ.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Quickstart — Setup](#quickstart---setup)
- [Training](#training)
- [Balancing & Augmentation](#balancing--augmentation)
- [Evaluation](#evaluation)
- [Inference / Gradio Demo](#inference--gradio-demo)
- [Tips & Troubleshooting](#tips--troubleshooting)
- [License & Acknowledgements](#license--acknowledgements)

---

## Project Overview

This project trains a ResNet50 model on the HAM10000 dataset (dermatoscopy images) to predict seven lesion types. The notebook implements:

- Class-specific balancing by repeated sampling (multipliers per class).
- Data augmentation (flips, rotations, color jitter) for training.
- Training loop with early stopping, validation, and test evaluation.
- Confusion matrix & classification report visualization.
- Saving the trained model (`skin_cancer_model.pth`) and a lightweight Gradio interface for inference.

The model architecture used: ResNet50 (final fully connected layer replaced to output 7 classes).

---

## Repository Structure (example)

(Adjust to match actual repository files)

- `notebook.ipynb` — main training & EDA notebook (contains all training code shown)
- `train.py` — optional training script (if present)
- `inference.py` — Gradio demo script (loads `skin_cancer_model.pth` and runs predictions)
- `skin_cancer_model.pth` — saved PyTorch model weights (generated after training)
- `requirements.txt` — Python dependencies
- `README.md` — this file

---

## Requirements

Recommended environment:

- Python 3.8+
- PyTorch (1.12+ recommended; supports CUDA if using GPU)
- torchvision
- pillow (PIL)
- numpy, pandas, matplotlib, seaborn
- scikit-learn
- tqdm
- gradio

Install via pip:

```bash
pip install torch torchvision pillow numpy pandas matplotlib seaborn scikit-learn tqdm gradio
```

If you have a CUDA-enabled GPU, install the matching PyTorch build from https://pytorch.org/get-started/locally.

---

## Dataset

This project uses the HAM10000 dataset. Expected layout (example):

dataset/
├─ HAM10000_metadata.csv
├─ HAM10000_images_part_1/
│  ├─ ISIC_0000000.jpg
│  └─ ...
└─ HAM10000_images_part_2/
   ├─ ISIC_0001000.jpg
   └─ ...

The notebook reads metadata and maps `image_id` to actual image paths. Update `dataset_path` in the notebook or script to point to your dataset root.

Important columns in metadata: `image_id`, `dx` (diagnosis label), `age`, `sex`, `localization`.

---

## Quickstart — Setup

1. Clone the repository
2. Install dependencies (see Requirements)
3. Place the HAM10000 dataset in your machine and update `dataset_path` variable in the notebook/script.
4. (Optional) Create a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

---

## Training

You can run the training from the notebook or move training code into a `train.py` script.

Important configuration points from the notebook:

- Input size: 224×224
- Model: `models.resnet50(weights=models.ResNet50_Weights.DEFAULT)` with `model.fc = nn.Linear(..., 7)`
- Criterion: `CrossEntropyLoss()`
- Optimizer: `Adam(model.parameters(), lr=0.001)`
- Early stopping: patience default used in the notebook (e.g. 10)
- Save best model: `torch.save(model.state_dict(), 'skin_cancer_model.pth')`

Example (notebook): run all cells to train. If you create `train.py`, structure it to accept CLI args (dataset path, epochs, batch size).

---

## Balancing & Augmentation

The notebook demonstrates class balancing by repeating rows per-class with class-specific multipliers:

Example multipliers used:
- `data_aug_rate = [15, 10, 5, 50, 5, 1, 40]` for classes `[akiec, bcc, bkl, df, mel, nv, vasc]`

Training augmentations include:
- RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(20)
- ColorJitter(brightness=0.1, contrast=0.1, hue=0.1)

A validation/test transform uses deterministic resizing and normalization.

---

## Evaluation

The notebook computes:
- Validation and test loss & accuracy
- Confusion matrix (plotted with seaborn)
- Classification report (precision/recall/F1)
- Training & validation loss/accuracy curves

Save the best model by validation loss and reload it for test evaluation.

---

## Inference / Gradio Demo

A Gradio interface is provided to upload an image and get:
- Predicted class label
- Whether the predicted class is considered cancerous (quick heuristic)
- Class probability scores (formatted percentages)

To run the Gradio app (example):

```bash
python inference.py
# or run the demo cell in the notebook
```

Key expectations:
- The script expects `skin_cancer_model.pth` to be in the working directory (or update the path).
- Use the same validation transforms (resize, normalize) used during training.

---

## Tips & Troubleshooting

- GPU: Training is much faster on CUDA. Check available device via `torch.cuda.is_available()`.
- DataLoader workers: On Windows, set `num_workers=0` to avoid hanging; other OSes can use >0.
- Memory: If you run out of GPU memory, reduce batch size (e.g., 64 → 32).
- Reproducibility: Set random seeds for numpy, torch, and python `random` to improve reproducibility.
- If you get different results: vary learning rate, data augmentation, or balancing multipliers.

---

## Reproducibility & Notes

- The notebook uses repeated-sampling balancing (simple replication). Alternatives: weighted sampling, SMOTE, or class-aware augmentations.
- The notebook reports high accuracy in demonstration runs; however, real-world performance depends on train/val/test splits, augmentation, and hyperparameter tuning.

---

## License & Acknowledgements

- HAM10000 dataset: see original dataset license and citation requirements.
- This code is provided as-is for educational/demo purposes.

If you'd like, I can also:
- Produce a minimal `requirements.txt`.
- Extract the training cells into a runnable `train.py`.
- Create an `inference.py` script that exactly matches the Gradio demo code you provided.

Contact: @Mohd-shafiq (GitHub)
