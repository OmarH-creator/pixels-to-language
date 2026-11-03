# Simple Grad-CAM: 8 correct and 8 wrong predictions.
# Run from the project folder: C:\miniconda3\python.exe gradcam.py
import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from src.models.baseline_cnn import BaselineCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Validation images only, so the test set stays untouched
dataset = datasets.ImageFolder("classification_data_square_288/val", transform=transforms.ToTensor())
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

model = BaselineCNN(num_classes=200).to(device)
model.load_state_dict(torch.load("outputs/square_288_400_wd005/best.pt", map_location=device, weights_only=True))
model.eval()


# Every time the model runs, keep the output of stage 5 (the last convolution stage)
def save_maps(module, inputs, output):
    global maps
    output.retain_grad()
    maps = output


model.stage5.register_forward_hook(save_maps)

correct = []
wrong = []
random.seed(42)
for index in random.sample(range(len(dataset)), len(dataset)):
    if len(correct) == 8 and len(wrong) == 8:
        break

    # Ask the model for its answer
    image, label = dataset[index]
    scores = model(normalize(image).unsqueeze(0).to(device))
    predicted = scores.argmax().item()

    group = correct if predicted == label else wrong
    if len(group) == 8:
        continue

    # Find how much each feature map affects the predicted species
    model.zero_grad()
    scores[0, predicted].backward()
    weights = maps.grad.mean(dim=(2, 3), keepdim=True)

    # Add up the maps using those weights, then enlarge the result to the image size
    heatmap = F.relu((weights * maps).sum(dim=1, keepdim=True))
    heatmap = F.interpolate(heatmap, size=(288, 288), mode="bilinear", align_corners=False)
    group.append((image, heatmap[0, 0].detach().cpu(), label, predicted))

# Top two rows: correct. Bottom two rows: wrong. Red = most important.
fig, axes = plt.subplots(4, 4, figsize=(16, 18))
for axis, (image, heatmap, label, predicted) in zip(axes.flat, correct + wrong):
    axis.imshow(image.permute(1, 2, 0))
    axis.imshow(heatmap, cmap="jet", alpha=0.45)
    result = "Correct" if predicted == label else "Wrong"
    axis.set_title(f"{result}\nTrue: {dataset.classes[label][4:]}\nPredicted: {dataset.classes[predicted][4:]}", fontsize=9)
    axis.axis("off")

os.makedirs("outputs/gradcam", exist_ok=True)
plt.tight_layout()
plt.savefig("outputs/gradcam/gradcam_16.png", dpi=150)
print("Saved: outputs/gradcam/gradcam_16.png")
