# From Pixels to Language
An end-to-end multimodal deep learning system built on the **CUB-200-2011** birds dataset. A shared visual CNN encoder backbone extracts spatial features for species classification (with Grad-CAM explainability), reconstructs degraded visual data via image colorization, and autoregressively generates natural language captions.

---

## 1. Environment Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- NVIDIA GPU with CUDA support recommended (CPU is fully supported as fallback)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd pixels-to-language
   ```

2. **Install PyTorch:**
   - **For NVIDIA GPU (CUDA acceleration):**
     ```bash
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
     ```
   - **For CPU-only:**
     ```bash
     pip install torch torchvision
     ```

3. **Install project dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import torch; print('PyTorch Version:', torch.__version__, '| CUDA Available:', torch.cuda.is_available())"
   ```

---

## 2. Dataset Preparation

1. Download the **CUB-200-2011** dataset from the official source or Hugging Face.
2. Place the unzipped dataset folder named `CUB_200_2011` directly into the repository root:
   ```text
   pixels-to-language/
   └── CUB_200_2011/
       ├── images/
       ├── images.txt
       └── train_test_split.txt
   ```
3. Run the data preprocessing pipeline:
   ```bash
   python -m src.data.preprocess
   ```
   This resizes images to $224 \times 224$, removes known duplicates, allocates a balanced 6-image-per-species validation split, and outputs the directory structure:
   ```text
   classification_data/
   ├── train/   (4,793 images across 200 species)
   ├── val/     (1,200 images across 200 species)
   └── test/    (5,794 images across 200 species)
   ```

4. *(Optional — for captioning in Milestone 2)* Clean the captions:
   ```bash
   python -m src.data.clean_captions
   ```

---

## 3. Training the Custom CNN

The training engine automatically detects hardware (CUDA GPU or CPU), logs per-epoch training and validation loss/accuracy, manages model checkpoints based on validation accuracy records, and generates learning curves.

Run training with:
```bash
python -m src.training.train
```

### Outputs Produced:
- **Best Model Checkpoint:** `outputs/checkpoints/baseline_v3_resnet18_best.pth`
- **Training & Validation Curves:** `outputs/training_curves/exp6_v3_resnet18_25ep.png`

---

## 4. Model Explainability (Grad-CAM)

To inspect what features the model focuses on for predictions:

```bash
python -m src.visualization.gradcam
```

### How it works:
- Hooks into the final convolutional stage (`stage4` in V3 ResNet-18) to extract $7 \times 7$ activation maps.
- Computes class-specific gradients via backpropagation.
- Randomly samples the validation set across all 200 bird species to find both **CORRECT** and **WRONG** predictions.
- Saves a timestamped 3-column visualization grid:
  `[Input Bird Image] | [Grad-CAM Activation Map] | [Superimposed Overlay]`
- **Saved Location:** `outputs/explainability_gradcam/gradcam_grid_YYYYMMDD_HHMMSS.png`

---

## 5. Evaluation & Transfer Learning

### Evaluate the From-Scratch Model:
```bash
python -m src.evaluation.Evaluate_test
python -m src.evaluation.Evaluate_nobox
```

### Transfer-Learning Classification (ResNet-18 / 34 / 50):
Train a pretrained ResNet using the CLI (`configs/cls_transfer.yaml` chooses `resnet18`, `resnet34`, or `resnet50`):
```bash
python -m src.cli train --task classification --config configs/cls_transfer.yaml
```

Predict a single image with a saved checkpoint:
```bash
python -m src.cli predict --task classification --image path/to/bird.jpg --checkpoint outputs/classification/resnet18/best.pt
```

---

## 6. Model Architecture Summary

| Model File | Class Name | Architecture Summary | Parameters |
| :--- | :--- | :--- | :--- |
| `src/models/baseline_cnn_v1.py` | `BaselineCNN_V1` | 4-Stage plain CNN, single conv per stage, MaxPool | ~440K |
| `src/models/baseline_cnn_v2_residual.py` | `BaselineCNN_V2_Residual` | 5-Stage residual CNN, double convs per stage, 512 max channels | ~5.0M |
| `src/models/baseline_cnn_v3_resnet18.py` | `BaselineCNN_V3_ResNet18` | 18 weighted layers, ResNet-18 basic blocks, stride-2 downsampling | ~2.85M |
| `src/models/baseline_cnn.py` | `BaselineCNN` | 25-Layer 5-Stage Residual CNN (`[2,2,3,3,2]` blocks) + Spatial Attention Pooling | ~12.8M |
| `src/models/transfer.py` | `build_model` | ImageNet-pretrained ResNet-18 / 34 / 50 with 200-class head | ~11.3M (ResNet-18) |

---

## 7. Repository Layout

```text
pixels-to-language/
├── configs/                                  # Hyperparameters & run settings (cls_transfer.yaml)
├── notebooks/                                # Exploratory notebooks (EDA.ipynb, transfer_learning.ipynb)
├── outputs/                                  # Experiment artifacts, logs & plots
├── src/
│   ├── data/                                 # Preprocessing and dataset pipelines
│   │   ├── preprocess.py
│   │   ├── clean_captions.py
│   │   └── dataset.py
│   ├── evaluation/                           # Evaluation & metrics scripts
│   │   ├── Evaluate_test.py
│   │   ├── Evaluate_nobox.py
│   │   └── metrics.py
│   ├── models/                               # From-scratch & transfer learning architectures
│   │   ├── baseline_cnn.py
│   │   ├── baseline_cnn_v1.py
│   │   ├── baseline_cnn_v2_residual.py
│   │   ├── baseline_cnn_v3_resnet18.py
│   │   └── transfer.py
│   ├── training/                             # Training loops
│   │   ├── train.py
│   │   └── trainer.py
│   ├── visualization/                        # Explainability & plotting tools
│   │   ├── gradcam.py
│   │   └── curves.py
│   └── cli.py                                # Command line interface entry point
├── PREPROCESSING.md                          # Data & caption preprocessing guide
├── requirements.txt                          # Python package dependencies
└── README.md                                 # Environment setup & replication guide
```

