import os
import random
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from src.models.baseline_cnn_v3_resnet18 import BaselineCNN_V3_ResNet18


class GradCAM:
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM)
    for a chosen target layer (Stage 5 in our CNN).
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, target_class=None):
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # Channel importance weights from average gradients
        weights = torch.mean(self.gradients[0], dim=(1, 2))
        activations = self.activations[0]
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        return cam.squeeze().cpu().numpy(), target_class, logits


def run_randomized_gradcam(model, target_layer, class_names, device, num_correct=2, num_wrong=2, save_name="gradcam_random_grid.png"):
    """
    Shuffles and scans the validation set randomly across all 200 species
    to visualize diverse birds each run.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dir = "classification_data/val"
    all_image_paths = []

    # Gather all 1,200 validation images
    for root, dirs, files in os.walk(val_dir):
        for f in files:
            if f.endswith((".jpg", ".png", ".jpeg")):
                all_image_paths.append(os.path.join(root, f))

    # Shuffle them so every run analyzes different species!
    random.shuffle(all_image_paths)

    correct_samples = []
    wrong_samples = []

    print(f"[Grad-CAM] Scanning randomly through validation images...")

    for img_path in all_image_paths:
        # Extract folder species name
        true_species = os.path.basename(os.path.dirname(img_path))
        true_idx = class_names.index(true_species)

        raw_img = Image.open(img_path).convert("RGB").resize((224, 224))
        inp_tensor = transform(raw_img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(inp_tensor)
            pred_idx = torch.argmax(out, dim=1).item()

        if pred_idx == true_idx and len(correct_samples) < num_correct:
            correct_samples.append({
                "path": img_path, "raw": raw_img, "tensor": inp_tensor,
                "true_name": true_species, "pred_name": class_names[pred_idx],
                "status": "CORRECT"
            })
        elif pred_idx != true_idx and len(wrong_samples) < num_wrong:
            wrong_samples.append({
                "path": img_path, "raw": raw_img, "tensor": inp_tensor,
                "true_name": true_species, "pred_name": class_names[pred_idx],
                "status": "WRONG"
            })

        if len(correct_samples) >= num_correct and len(wrong_samples) >= num_wrong:
            break

    all_samples = correct_samples + wrong_samples
    total_imgs = len(all_samples)

    if total_imgs == 0:
        print("Error: Could not find any valid samples.")
        return

    cam_engine = GradCAM(model, target_layer)

    fig, axes = plt.subplots(total_imgs, 3, figsize=(14, 4 * total_imgs))
    if total_imgs == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, s in enumerate(all_samples):
        heatmap, pred_idx, _ = cam_engine.generate_heatmap(s["tensor"])

        status_color = "#16a34a" if s["status"] == "CORRECT" else "#dc2626"
        title_prefix = f"[{s['status']}]"

        # 1. Input image
        axes[i, 0].imshow(s["raw"])
        axes[i, 0].set_title(f"{title_prefix} True: {s['true_name']}\nPred: {s['pred_name']}",fontsize=10, fontweight="bold", color=status_color)
        axes[i, 0].axis("off")

        # 2. Heatmap
        axes[i, 1].imshow(heatmap, cmap="jet")
        axes[i, 1].set_title("Grad-CAM Activation Map", fontsize=10, fontweight="bold")
        axes[i, 1].axis("off")

        # 3. Superimposed Overlay
        axes[i, 2].imshow(s["raw"])
        axes[i, 2].imshow(heatmap, cmap="jet", alpha=0.45)
        axes[i, 2].set_title("Overlay (Attention Focus)", fontsize=10, fontweight="bold")
        axes[i, 2].axis("off")

    plt.tight_layout()
    out_dir = "outputs/explainability_gradcam"
    os.makedirs(out_dir, exist_ok=True)

    # Use timestamp so every run is saved distinctly and never overwritten
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = f"gradcam_grid_{timestamp}.png"
    out_file = os.path.join(out_dir, out_filename)

    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"\n[Grad-CAM] Successfully saved randomized analysis to: {out_file}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "outputs/checkpoints/baseline_v3_resnet18_best.pth"

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint '{checkpoint_path}' not found! Run train.py first.")
        exit(1)

    print(f"Loading trained weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("classes", None)

    if class_names is None:
        val_dir = "classification_data/val"
        class_names = sorted([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))])

    num_classes = len(class_names)
    model = BaselineCNN_V3_ResNet18(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    target_layer = model.stage4

    # Picks 2 random correct birds and 2 random wrong birds across all 200 species
    run_randomized_gradcam(model, target_layer, class_names, device, num_correct=2, num_wrong=2, save_name="gradcam_random_grid.png")