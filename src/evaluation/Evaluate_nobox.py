from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.models.baseline_cnn import BaselineCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
finish = [transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
square_val = Path("classification_data_square_288/val")
raw = Path("CUB_200_2011/images")

# Each test: the folder to read from, and how to prepare the image
tests = {
    "With boxes (should be near 69.58%)": (square_val, transforms.Compose(finish)),
    "Full photo, centre square": (raw, transforms.Compose(
        [transforms.Resize(288), transforms.CenterCrop(288)] + finish)),
    "Full photo, squeezed": (raw, transforms.Compose(
        [transforms.Resize((288, 288))] + finish)),
}

model = BaselineCNN(num_classes=200).to(device)
model.load_state_dict(torch.load("outputs/square_288_400_wd005/best.pt", map_location=device, weights_only=True))
model.eval()

species = sorted(folder.name for folder in square_val.iterdir())
files = [(path.relative_to(square_val), label)
         for label, name in enumerate(species)
         for path in sorted((square_val / name).glob("*.jpg"))]

with torch.no_grad():
    for title, (folder, prepare) in tests.items():
        correct = 0
        for path, label in files:
            with Image.open(folder / path) as image:
                x = prepare(image.convert("RGB")).unsqueeze(0).to(device)
            correct += model(x).argmax(dim=1).item() == label
        print(f"{title}: {100 * correct / len(files):.2f}%")