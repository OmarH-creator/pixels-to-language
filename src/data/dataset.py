# Reads classification_data from src.data.preprocess.
# Species folder names are the labels. Images are already 224x224 RGB JPEGs.
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def make_transform(train):
    steps = [transforms.Resize((224, 224))]
    if train:
        steps.append(transforms.RandomHorizontalFlip())
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(MEAN, STD))
    return transforms.Compose(steps)


class BirdDataset(Dataset):
    def __init__(self, root, class_names, train):
        self.transform = make_transform(train)
        labels = {name: index for index, name in enumerate(class_names)}
        self.samples = []
        for species in class_names:
            folder = Path(root) / species
            if not folder.is_dir():
                continue
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() == ".jpg":
                    self.samples.append((path, labels[species]))
        if not self.samples:
            raise ValueError("No JPEG images in " + str(root))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label


def classification_loaders(data_dir, batch_size):
    data_dir = Path(data_dir)
    if not (data_dir / "complete.txt").is_file():
        raise ValueError("Run python -m src.data.preprocess first.")

    names = set()
    for split in ("train", "val", "test"):
        folder = data_dir / split
        if not folder.is_dir():
            raise ValueError("Missing split folder: " + str(folder))
        for species in folder.iterdir():
            if species.is_dir():
                names.add(species.name)
    class_names = sorted(names)
    if not class_names:
        raise ValueError("No species folders in " + str(data_dir))

    datasets = {
        "train": BirdDataset(data_dir / "train", class_names, train=True),
        "val": BirdDataset(data_dir / "val", class_names, train=False),
        "test": BirdDataset(data_dir / "test", class_names, train=False),
    }
    for split, dataset in datasets.items():
        print(split, "images:", len(dataset), flush=True)

    loaders = {}
    for split, dataset in datasets.items():
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
        )
    return loaders, class_names
