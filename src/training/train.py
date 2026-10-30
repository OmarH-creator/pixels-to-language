import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Using our 18-layer ResNet-style architecture (V3)
from src.models.baseline_cnn_v3_resnet18 import BaselineCNN_V3_ResNet18

# 1. Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Image Transformations
# Mild spatial augmentations (scale 0.8 to 1.0 keeps at least 80% of the image)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Load Datasets
train_dataset = datasets.ImageFolder("classification_data/train", transform=train_transform)
val_dataset   = datasets.ImageFolder("classification_data/val", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"Train samples: {len(train_dataset)} across {len(train_dataset.classes)} species.")
print(f"Val samples:   {len(val_dataset)} across {len(val_dataset.classes)} species.")

# 4. Initialize Model, Loss & Optimizer
num_classes = len(train_dataset.classes)
model = BaselineCNN_V3_ResNet18(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

num_epochs = 45
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

# -------------------------------------------------------------
# 5. Checkpoint Management: Fetch previous best if available
# -------------------------------------------------------------
os.makedirs("outputs/checkpoints", exist_ok=True)
checkpoint_path = "outputs/checkpoints/baseline_v3_resnet18_best.pth"

if os.path.exists(checkpoint_path):
    print(f"\n[Checkpoint] Found existing saved model at: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    best_val_acc = checkpoint.get("best_val_acc", 0.0)
    saved_epoch = checkpoint.get("epoch", 0)
    print(f"[Checkpoint] Historical best to beat: {best_val_acc:.2f}% (achieved at epoch {saved_epoch})\n")
else:
    best_val_acc = 0.0
    print(f"\n[Checkpoint] No previous checkpoint found. Best record set to: 0.00%\n")

# 6. Training & Validation Loop
history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

print("Starting Training...")
print("=" * 65)

for epoch in range(1, num_epochs + 1):
    # A. Training Phase
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = torch.argmax(outputs, dim=1)
        correct_train += (predictions == labels).sum().item()
        total_train += labels.size(0)

    epoch_train_loss = running_loss / total_train
    epoch_train_acc = (correct_train / total_train) * 100.0

    # B. Validation Phase
    model.eval()
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
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

    scheduler.step()

    history["train_loss"].append(epoch_train_loss)
    history["train_acc"].append(epoch_train_acc)
    history["val_loss"].append(epoch_val_loss)
    history["val_acc"].append(epoch_val_acc)

    print(f"Epoch [{epoch:02d}/{num_epochs:02d}] | "
          f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.2f}% | "
          f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%", end="")

    # Checkpoint saving: update disk ONLY when beating the all-time best score
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "classes": train_dataset.classes
        }, checkpoint_path)
        print(f"  --> [NEW RECORD! Saved {best_val_acc:.2f}%]")
    else:
        print()

# 7. Plot Curves
print("\nTraining complete! Generating visualization plot...")
os.makedirs("outputs/training_curves", exist_ok=True)
plot_path = "outputs/training_curves/exp6_v3_resnet18_25ep.png"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.plot(range(1, num_epochs + 1), history["train_loss"], label="Train Loss", marker="o", color="#2563eb")
ax1.plot(range(1, num_epochs + 1), history["val_loss"], label="Val Loss", marker="o", color="#f97316")
ax1.set_title("Loss vs. Epochs", fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("CrossEntropy Loss")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.5)

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