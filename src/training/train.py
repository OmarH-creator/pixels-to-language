import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from src.models.baseline_cnn import BaselineCNN

torch.manual_seed(42)

# 1. Device Setup
# If you have an NVIDIA GPU set up with CUDA, it uses GPU; otherwise CPU
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Image Transformations
# These tell PyTorch how to prepare each raw image before feeding it to the model
# Training gets random augmentations so the model learns general bird features
train_transform = transforms.Compose([
    ##########################################################################
    # BIGGER IMAGES: crop size matches the 288 images
    transforms.RandomResizedCrop(288, scale=(0.7, 1.0)),
    ##########################################################################
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25),
])
# Validation stays clean (no random flips or crops during the exam!)
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# 3. Load the Data using ImageFolder
##########################################################################
# SQUARE CROP: use the 288 square images made by preprocess.py
train_dataset = datasets.ImageFolder("classification_data_square_288/train", transform=train_transform)
val_dataset   = datasets.ImageFolder("classification_data_square_288/val", transform=val_transform)
##########################################################################

# 4. DataLoaders (The batch dispensers)
# Neural networks don't look at 1 image at a time (too slow),
# nor all 6,000 at once (GPU runs out of memory).
# They look at a 'batch' (e.g. 16 or 32 at a time).
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, persistent_workers=True, num_workers=8)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, persistent_workers=True, num_workers=8)

print(f"Train samples: {len(train_dataset)} across {len(train_dataset.classes)} species.")
print(f"Val samples:   {len(val_dataset)} across {len(val_dataset.classes)} species.")

# //////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////

# 5. Initialize the Model
# We tell it there are 200 bird species, and send its weights to our device (GPU or CPU)
num_classes = len(train_dataset.classes)
model = BaselineCNN(num_classes=num_classes).to(device)

# 6. Loss Function
# CrossEntropyLoss does two things:
#   a) Applies Softmax to turn the 200 raw scores into percentages/probabilities
#   b) Measures how far the predicted bird is from the true bird label
criterion = nn.CrossEntropyLoss()

##########################################################################
# CUTMIX: pastes part of one image onto another and mixes their labels
from torchvision.transforms import v2
cutmix = v2.CutMix(num_classes=num_classes)
##########################################################################

# 7. Optimizer
# AdamW is an improved version of Adam with proper weight decay (L2 regularization).
# lr=0.001 is the step size.
##########################################################################
# WEIGHT DECAY: raised from 1e-4 to 0.05, so weights stay small and memorizing is harder
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
##########################################################################
##########################################################################
# Longer run, because val loss was still falling at 100 epochs
num_epochs = 400
##########################################################################

# 7b. Learning Rate Scheduler
# Smoothly cools down the learning rate from 1e-3 to 1e-5 over the epochs
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

# //////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////



# 8. Lists to record history for our visualization curves
history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}


print("\nStarting Training...")
print("=" * 65)

os.makedirs("outputs/baseline", exist_ok=True)
best_val_acc = 0.0

##########################################################################
# MIXED PRECISION: only on the GPU. The scaler stops very small gradients becoming zero
use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
##########################################################################

for epoch in range(1, num_epochs + 1):
    # ---------------------------
    # A. TRAINING PHASE
    # ---------------------------
    model.train()  # Tell layers like Dropout & BatchNorm that we are training
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        ##########################################################################
        # CUTMIX: keep the real labels for accuracy, then mix the batch
        original_labels = labels
        images, labels = cutmix(images, labels)
        ##########################################################################

        # 1. Clear old gradients
        optimizer.zero_grad()

        ##########################################################################
        # MIXED PRECISION: the forward pass uses lower-precision numbers where it is safe
        # 2. Forward pass: Model makes predictions
        with torch.autocast(device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # 3. Backward pass: Calculate gradients
        scaler.scale(loss).backward()

        # 4. Optimizer: Update weights
        scaler.step(optimizer)
        scaler.update()
        ##########################################################################

        # Track accuracy and loss
        running_loss += loss.item() * images.size(0)
        predictions = torch.argmax(outputs, dim=1)
        ##########################################################################
        # CUTMIX: labels are mixed now, so compare with the real labels
        correct_train += (predictions == original_labels).sum().item()
        ##########################################################################
        total_train += labels.size(0)

    epoch_train_loss = running_loss / total_train
    epoch_train_acc = (correct_train / total_train) * 100.0

    # ---------------------------
    # B. VALIDATION PHASE (The Exam)
    # ---------------------------
    model.eval()  # Freeze BatchNorm stats and turn off Dropout
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():  # Do NOT calculate gradients during validation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            predictions = torch.argmax(outputs, dim=1)
            correct_val += (predictions == labels).sum().item()
            total_val += labels.size(0)

    epoch_val_loss = val_running_loss / total_val
    epoch_val_acc = (correct_val / total_val) * 100.0

        # Update the learning rate for the next epoch
    scheduler.step()
    # Save to history
    history["train_loss"].append(epoch_train_loss)
    history["train_acc"].append(epoch_train_acc)
    history["val_loss"].append(epoch_val_loss)
    history["val_acc"].append(epoch_val_acc)

    print(f"Epoch [{epoch:02d}/{num_epochs:02d}] | "
        f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.2f}% | "
        f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%")

    with open("outputs/baseline/history.json", "w") as file:
        json.dump(history, file, indent=2)
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), "outputs/baseline/best.pt")

# //////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////

# 9. Plot and Save Learning Curves
print("\nTraining complete! Generating visualization plot...")
os.makedirs("outputs", exist_ok=True)
plot_path = "outputs/training_curves.png"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Loss Curve
ax1.plot(range(1, num_epochs + 1), history["train_loss"], label="Train Loss", marker="o", color="#2563eb")
ax1.plot(range(1, num_epochs + 1), history["val_loss"], label="Val Loss", marker="o", color="#f97316")
ax1.set_title("Loss vs. Epochs", fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("CrossEntropy Loss")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.5)

# Accuracy Curve
ax2.plot(range(1, num_epochs + 1), history["train_acc"], label="Train Acc", marker="s", color="#16a34a")
ax2.plot(range(1, num_epochs + 1), history["val_acc"], label="Val Acc", marker="s", color="#9333ea")
ax2.set_title("Accuracy vs. Epochs", fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Curves saved to: {plot_path}")