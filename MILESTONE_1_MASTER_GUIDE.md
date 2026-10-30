# CSE 485 / CSE 488: Deep Learning — Milestone 1 Comprehensive Master Guide
## "From Pixels to Language: CNN Encoder Backbone, Training Engine & Explainability"

---

## 1. Project Overview & The Unifying Philosophy

### 1.1 The High-Level Goal
This project is an end-to-end multimodal deep learning system built around the **CUB-200-2011 (Caltech-UCSD Birds)** dataset. The dataset consists of **11,788 bird photographs across 200 distinct bird species**, accompanied by natural language human descriptions (10 captions per image).

The core technical philosophy of this entire project is **The Shared Visual Backbone**:
```
                           ┌──► Task 1: Image Classification (Species ID) [Milestone 1]
                           │
Raw Bird Image ──► CNN Encoder Backbone ──► Task 2: Image Colorization (Grayscale -> Color U-Net) [Milestone 2]
                           │
                           └──► Task 3: Image Captioning (Visual Features -> Autoregressive Text) [Milestone 2]
```
What you build and train in Milestone 1 acts as the visual feature extractor (the "eyes") for everything that comes in Milestone 2.

### 1.2 Grading Allocation & The Significance of Technical Understanding
* **Total Marks:** 20 marks (CSE 488) / 25 marks (CSE 485).
* **Individual Oral Discussion (35%):** The single largest grading component. Each student is interviewed individually on the engineering decisions, trade-offs, failure modes, and mathematical mechanics of the entire system.
* **Milestone 1 Deliverables (Task 1):**
  1. Exploratory Data Analysis (EDA) and preprocessed folder structure.
  2. Training a Custom CNN from scratch (the baseline).
  3. Transfer Learning comparison (ResNet/EfficientNet).
  4. Hyperparameter tuning (comparing Grid Search vs. Random Search).
  5. Model Explainability via Grad-CAM (identifying what the model looks at for correct and wrong predictions).

---

## 2. Team Division Strategy (3 Members)

To ensure meaningful individual GitHub contributions and prevent the trap where members cannot defend parts outside their own focus during the 35% oral discussion:

```
┌─────────────────────────────────┬─────────────────────────────────┬─────────────────────────────────┐
│ Member 1: Data & Preprocessing  │ Member 2: Baseline Engine (YOU) │ Member 3: Transfer & Tuning     │
├─────────────────────────────────┼─────────────────────────────────┼─────────────────────────────────┤
│ • Dataset download & extraction │ • Custom CNN Architectures:     │ • Transfer Learning models:     │
│ • Caption cleaning & tokenizing │   - BaselineCNN_V1 (4-stage)    │   - ResNet50 / ConvNeXt         │
│ • Preprocessing script          │   - BaselineCNN_V2 (Residual)   │ • Hyperparameter Optimization:  │
│   (classification_data/ splits) │   - BaselineCNN_V3 (ResNet-18)  │   - Grid Search vs Random Search│
│ • EDA statistical plots         │ • PyTorch Training Engine:      │ • Open-source benchmarking      │
│ • Lead on Task 3 (Milestone 2)  │   - Checkpoint dictionary manager│ • Benchmarking comparisons      │
│                                 │   - Ablation experiment suite   │ • Lead on Benchmarking (M2)     │
│                                 │ • Explainability via Grad-CAM   │                                 │
│                                 │ • Lead on Task 2 (Milestone 2)  │                                 │
└─────────────────────────────────┴─────────────────────────────────┴─────────────────────────────────┘
```

---

## 3. The Dataset Reality: CUB-200-2011

Before writing network code, understanding the mathematical characteristics of the dataset is essential:
* **Total Images:** 11,788.
* **Number of Classes:** 200 species.
* **Official Split:** 5,994 Train / 5,794 Test.
* **Cleaned Split (Member 1's `preprocess.py`):**
  - Removed 1 duplicate image (`image_id == 3340`).
  - Allocated $6 \text{ images} \times 200 \text{ species} = 1,200 \text{ images}$ for Validation.
  - Final training set: **4,793 images**.
* **The Fine-Grained Scarcity Challenge:**
  $$\frac{4,793 \text{ training images}}{200 \text{ species}} \approx \mathbf{24 \text{ images per species}}$$
  This data scarcity is the defining constraint of the project. A large model can easily memorize 24 images, causing severe overfitting. Every architectural and regularization choice must counteract this reality.

---

## 4. Fundamental Deep Learning Mathematics & Mechanics

### 4.1 How Images Flow Through PyTorch: The 4D Tensor
Every image batch is represented as a 4-dimensional tensor with shape:
$$\text{Shape: } [B, C, H, W]$$
* **$B$ (Batch Size):** Number of images processed simultaneously (e.g., 16).
* **$C$ (Channels):** 3 color channels (Red, Green, Blue).
* **$H, W$ (Spatial Dimensions):** Height and Width ($224 \times 224$ pixels).

### 4.2 Parameter Counting Formulas
During oral exams, calculating parameters by hand demonstrates core competence:

1. **Convolutional Layer (`nn.Conv2d` without bias):**
   $$\text{Params} = K_h \times K_w \times C_{\text{in}} \times C_{\text{out}}$$
   *Example: Conv2d(3, 32, kernel_size=3):* $3 \times 3 \times 3 \times 32 = 864$.

2. **Batch Normalization (`nn.BatchNorm2d`):**
   $$\text{Params} = 2 \times C_{\text{out}}$$
   Learns a scale factor ($\gamma$) and shift factor ($\beta$) for each channel.

3. **Linear Layer (`nn.Linear` with bias):**
   $$\text{Params} = (D_{\text{in}} \times D_{\text{out}}) + D_{\text{out}}$$

4. **Pooling Layers (`MaxPool2d`, `AdaptiveAvgPool2d`):**
   $$\text{Params} = 0$$
   Fixed deterministic mathematical operations; zero learnable parameters.

### 4.3 Why Global Average Pooling (GAP) is Critical
* **Legacy Approach (VGG-style Flattening):**
  Taking a $14 \times 14 \times 256$ tensor, flattening it into $50,176$ values, and connecting to 200 classes requires:
  $$50,176 \times 200 \approx \mathbf{10,035,200 \text{ parameters}}$$
  With only 24 images per class, 10 million parameters causes instant memorization (overfitting).
* **Modern Approach (Global Average Pooling):**
  Averages each spatial $14 \times 14$ grid down to a single value ($1 \times 1$). Tensor reduces from $[B, 256, 14, 14] \rightarrow [B, 256]$.
  $$\text{Parameters: } 256 \times 200 = \mathbf{51,200 \text{ parameters (99.5% reduction)}}$$

### 4.4 Theoretical Initial Loss
For an untrained model with random weights predicting uniformly across $K = 200$ classes:
$$\mathcal{L}_{\text{init}} = -\ln\left(\frac{1}{K}\right) = \ln(200) \approx \mathbf{5.298}$$
When your training script launched, Epoch 1 started at **5.34–5.43**, verifying mathematical correctness and unbiased initial predictions.

---

## 5. Architectural Evolution: V1 to V3

### 5.1 Architecture Comparison Table

| Property | Baseline V1 (`baseline_cnn_v1.py`) | Baseline V2 Residual (`baseline_cnn_v2_residual.py`) | Baseline V3 ResNet-18 (`baseline_cnn_v3_resnet18.py`) |
| :--- | :--- | :--- | :--- |
| **Philosophy** | Minimal sequential baseline | Custom residual downsampling | Standard ResNet-18 layout |
| **Weighted Layers** | 5 (4 Convs + 1 Linear) | 16 (10 main Convs + 5 Shortcuts + 1 Linear) | **18** (1 Stem + 16 Block Convs + 1 Linear) |
| **Convs per Stage** | 1 Conv per stage | 2 Convs + 1 Shortcut Conv | 2 BasicBlocks (4 Convs) per stage |
| **Downsampling** | MaxPool after every Conv | MaxPool after every block | **Stride-2 Convolutions** in first block |
| **Shortcuts** | None (pure plain CNN) | Always $1 \times 1$ Conv + BN | **Identity** when dimensions match; $1 \times 1$ only when sizing down |
| **Channel Progression**| $3 \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 256$ | $3 \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 256 \rightarrow 512$ | $3 \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 256$ |
| **Final Spatial Grid** | $14 \times 14$ | $7 \times 7$ | $7 \times 7$ |
| **Total Parameters** | **440,296** (~0.44M) | **4,992,968** (~5.0M) | **2,845,416** (~2.85M) |
| **Key Limitation** | Tiny receptive field; underfitting | Heavy parameters; pooling after every block | Deeper but lighter; stacks blocks at same resolution |

---

## 6. The Ablation Study Suite: Experiments 1 to 6

Every experiment was conducted scientifically by isolating variables to measure their individual impact:

```
Exp 1 (V1 Plain, 5 ep) ──► 4.25% Val Acc (Steep downward slope; Underfitting)
     │
Exp 2 (V1 + Aug + Sched, 15 ep) ──► 6.83% Val Acc (Capacity ceiling hit; Model brain too small)
     │
Exp 3 (V2 Residual No-Aug, 15 ep) ──► 17.17% Val Acc (Surge in capacity, but Train 25.7% vs Val 17.2% -> Overfitting gap)
     │
Exp 4 (V2 Residual + Aug, 15 ep) ──► 15.58% Val Acc (Gap closed: Train 18.6% vs Val 15.6%; Sched cut LR too early)
     │
Exp 5 (V2 Residual + Crop 0.5, 25 ep) ──► 22.67% Val Acc (Significant jump; Background shortcuts visually exposed)
     │
Exp 6 (V3 ResNet-18 + Crop 0.8, 25 ep) ──► 23.75% Val Acc (Peak Record; Perfect generalization, attention locks on birds!)
```

### 6.1 Comprehensive Ablation Results Table

| Exp # | Model Architecture | Training Strategy | Ep | Train Loss | Val Loss | Train Acc | Val Acc | Key Finding & Diagnosis |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | Baseline V1 (4-stage) | AdamW, No Aug, No Scheduler | 5 | 4.8313 | 4.8529 | 2.90% | **4.25%** | Underfitting. Verified loss initialization ($\approx 5.3$). |
| **2** | Baseline V1 (4-stage) | + Augmentation + Cosine Scheduler | 15 | 4.2878 | 4.5686 | 8.97% | **6.83%** | Flattens at ~7%. 4 Conv layers lack the capacity for 200 species. |
| **3** | Baseline V2 (Residual) | No Augmentation, No Scheduler | 15 | 3.0549 | 3.6337 | 25.73% | **17.17%** | Significant jump (+10.3%). Generalization gap opens (8.5% gap). |
| **4** | Baseline V2 (Residual) | + Augmentation + Cosine Scheduler | 15 | 3.4013 | 3.7739 | 18.65% | **15.58%** | Gap closes to 3.0%. Model learns general features; LR braked early. |
| **5** | Baseline V2 (Residual) | + RandomCrop(0.5, 1.0) + Cosine | 25 | 2.4850 | 3.3958 | 37.97% | **22.67%** | First run breaking 20%. Grad-CAM shows background water bias. |
| **6** | Baseline V3 (ResNet-18) | + RandomCrop(0.8, 1.0) + Cosine | 25 | 2.8938 | 3.2848 | 26.98% | **23.75%** | **All-time record.** Generalization curves align; Grad-CAM focuses on birds. |

---

## 7. Model Explainability via Grad-CAM

### 7.1 Mathematical Mechanism
Grad-CAM computes the gradient of the winning class score $y^c$ with respect to the feature map activations $A^k$ of the final convolutional stage:
$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
Where:
* $\alpha_k^c$ is the importance weight of channel $k$ for class $c$.
* $Z$ is the spatial area ($7 \times 7 = 49$).
* The final Class Activation Map is a weighted combination followed by ReLU:
  $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k} \alpha_k^c A^k\right)$$
* **Why ReLU?** We only care about visual features that positively contribute to the target class; negative features indicate counter-evidence.

### 7.2 Empirical Discoveries from Grad-CAM Grids

#### Discovery 1: "Clever Hans" Background Shortcut Learning (Exp 3 & Exp 5)
* In early models trained with wide crops, marine bird species (e.g., *Black-footed Albatross*, *Pacific Loon*) produced heatmaps highlighting **green frothy water ripples and ocean waves** while leaving the bird dark blue (zero attention).
* **Diagnosis:** The network found an easy statistical shortcut: recognizing water textures minimized training loss faster than learning subtle beak geometries across 24 images.

#### Discovery 2: Meaningful Anatomical Feature Learning (Exp 6)
With V3 ResNet-18 and mild cropping (`scale=(0.8, 1.0)`), the heatmaps demonstrate genuine anatomical feature detection:
* **Baltimore Oriole (`[CORRECT]`):** Hotspot centered on the bright orange breast and wing margin; background ignored.
* **European Goldfinch (`[CORRECT]`):** Heatmap focused on the distinct yellow-and-black wing bar.
* **Pied Kingfisher (`[CORRECT]`):** Attention isolated to the crested head and black-and-white collar.
* **Common Tern vs. Caspian Tern (`[WRONG]`):** Heatmap targeted the white neck, black crown, and bill. The network correctly identified the bird family by its facial anatomy, confusing sister species due to subtle beak color variation.

---

## 8. Directory & File Structure

```text
pixels-to-language/
├── configs/                                  # Hyperparameters & run settings
├── notebooks/                                # Scratch explorations only
├── outputs/                                  # Experiment artifacts & checkpoints
│   ├── checkpoints/
│   │   ├── baseline_v2_residual_5stage_best.pth  (Exp 5 weights, 22.67%)
│   │   └── baseline_v3_resnet18_best.pth         (Exp 6 weights, 23.75%)
│   ├── training_curves/
│   │   ├── exp1_v1_baseline_5ep_acc04.png
│   │   ├── exp2_v1_aug_scheduler_15ep_acc07.png
│   │   ├── exp3_v2_residual_noaug_15ep_acc17.png
│   │   ├── exp4_v2_residual_aug_15ep_acc16.png
│   │   ├── exp5_v2_residual_aug_crop_25ep_acc23.png
│   │   └── exp6_v3_resnet18_25ep.png
│   └── explainability_gradcam/
│       ├── gradcam_exp3_15ep_fixed_albatross.png  (Water shortcut visual evidence)
│       └── gradcam_grid_*.png                     (Timestamped randomized grids)
├── src/
│   ├── data/
│   │   ├── preprocess.py                     # Official CUB dataset preparation (Member 1)
│   │   └── clean_captions.py                 # Text description cleaning (Member 1)
│   ├── models/
│   │   ├── baseline_cnn_v1.py                # 4-stage minimal model (~440K params)
│   │   ├── baseline_cnn_v2_residual.py       # 5-stage residual model (~5.0M params)
│   │   └── baseline_cnn_v3_resnet18.py       # 18-layer standard ResNet (~2.85M params)
│   ├── training/
│   │   └── train.py                          # Unified training engine & checkpoint manager
│   ├── visualization/
│   │   └── gradcam.py                        # Randomized Grad-CAM visualizer
│   └── cli.py                                # CLI entry point
├── requirements.txt                          # Python dependencies
└── README.md                                 # Environment setup & replication commands
```

---

## 9. Key Questions for the 35% Oral Discussion

#### Q1: "Why did you build your own CNN from scratch if pre-trained ResNet gets 85%?"
> *"Building a CNN from scratch establishes an empirical baseline to measure the true benefit of Transfer Learning. It also reveals the challenges of learning representations under data scarcity (~24 images per class), such as vanishing gradients and background shortcut learning, which pre-trained weights circumvent."*

#### Q2: "Why does V3 ResNet-18 have FEWER parameters than V2 (2.85M vs 5.0M) despite having MORE layers (18 vs 10)?"
> *"Parameter count in CNNs is dominated by channel depth in late stages. V2 expanded to 512 channels in Stage 5 ($256 \times 512 \times 3 \times 3 \approx 1.18\text{M}$ per Conv). V3 caps channel depth at 256 and uses identity shortcuts where dimensions match. Depth and parameter count are separate design axes."*

#### Q3: "What is the difference between `model.train()`, `model.eval()`, and `torch.no_grad()`?"
> *"`model.train()` enables Dropout (randomly deactivating neurons) and sets BatchNorm to compute batch statistics. `model.eval()` disables Dropout and locks BatchNorm to use accumulated running mean and variance. `torch.no_grad()` is an autograd context manager that disables computation graph tracking, reducing memory consumption and speeding up validation inference."*

#### Q4: "Why did your early models focus on ocean water instead of birds in Grad-CAM?"
> *"Because background water textures are consistent across marine bird photos, providing an easy statistical shortcut to minimize cross-entropy loss. By implementing scale-controlled RandomResizedCrop (`scale=(0.8, 1.0)`), we reduced the reliability of background context, forcing the convolutions to focus on bird anatomy (beaks, crests, and plumage)."*

---

## 10. Next Steps for Milestone 1 Completion
1. **Coordinate with Member 3:** Benchmark V3 ResNet-18 (23.75%) against a pre-trained Transfer Learning model (e.g., pre-trained ResNet-50 / EfficientNet).
2. **Hyperparameter Tuning:** Set up Grid Search vs. Random Search over learning rates (`[1e-4, 5e-4, 1e-3]`) and weight decays (`[1e-4, 5e-4]`).
3. **CLI Packaging:** Wire `src/cli.py` to allow training execution via:
   ```powershell
   python -m src.cli train --task classification --config configs/cls.yaml
   ```
