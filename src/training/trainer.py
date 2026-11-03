# Fine-tune a pretrained ResNet. The saved model is the epoch with the
# highest validation macro F1. Test is scored once, after that choice.
import json
from pathlib import Path

import torch
from torch import nn

from src.data.dataset import classification_loaders, make_transform
from src.evaluation.metrics import classification_metrics
from src.models.transfer import build_model
from src.visualization.curves import save_training_curves

ROOT = Path(__file__).resolve().parents[2]


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train, num_classes):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    seen = 0
    predictions = []
    targets = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
        total_loss = total_loss + loss.item() * images.size(0)
        seen = seen + images.size(0)
        predictions.append(logits.argmax(1).detach().cpu())
        targets.append(labels.detach().cpu())

    metrics = classification_metrics(torch.cat(predictions), torch.cat(targets), num_classes)
    return total_loss / seen, metrics


def train_model(config):
    torch.manual_seed(42)
    device = pick_device()
    loaders, class_names = classification_loaders(
        ROOT / config["data_dir"],
        int(config["batch_size"]),
    )
    model = build_model(config["backbone"], len(class_names), pretrained=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(config["learning_rate"]),
        momentum=0.9,
    )
    output = ROOT / config["output_dir"] / config["backbone"]
    output.mkdir(parents=True, exist_ok=True)
    print("Device:", device, flush=True)
    print("Saving to:", output, flush=True)

    best_f1 = -1.0
    best_epoch = 0
    history = []
    num_classes = len(class_names)
    for epoch in range(1, int(config["epochs"]) + 1):
        train_loss, train_metrics = run_epoch(
            model, loaders["train"], criterion, optimizer, device, True, num_classes
        )
        val_loss, val_metrics = run_epoch(
            model, loaders["val"], criterion, optimizer, device, False, num_classes
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
            }
        )
        (output / "history.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            "epoch",
            epoch,
            "train loss",
            round(train_loss, 4),
            "acc",
            round(train_metrics["accuracy"], 4),
            "val loss",
            round(val_loss, 4),
            "acc",
            round(val_metrics["accuracy"], 4),
            "f1",
            round(val_metrics["f1_macro"], 4),
            flush=True,
        )
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "backbone": config["backbone"],
                    "class_names": class_names,
                    "epoch": epoch,
                },
                output / "best.pt",
            )
            (output / "val_metrics.json").write_text(
                json.dumps({"epoch": epoch, "loss": val_loss, **val_metrics}, indent=2) + "\n",
                encoding="utf-8",
            )

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_loss, test_metrics = run_epoch(
        model, loaders["test"], criterion, None, device, False, num_classes
    )
    (output / "test_metrics.json").write_text(
        json.dumps({"loss": test_loss, **test_metrics}, indent=2) + "\n",
        encoding="utf-8",
    )
    save_training_curves(history, output / "curves.png")
    text = "\n".join(
        [
            "backbone: " + config["backbone"],
            "best_epoch: " + str(best_epoch),
            "best_val_f1_macro: " + format(best_f1, ".6f"),
            "test_loss: " + format(test_loss, ".6f"),
            "test_accuracy: " + format(test_metrics["accuracy"], ".6f"),
            "test_precision_macro: " + format(test_metrics["precision_macro"], ".6f"),
            "test_recall_macro: " + format(test_metrics["recall_macro"], ".6f"),
            "test_f1_macro: " + format(test_metrics["f1_macro"], ".6f"),
            "Saved checkpoint: highest validation macro F1.",
            "",
        ]
    )
    (output / "results.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


def predict_image(image_path, checkpoint_path):
    device = pick_device()
    image_path = Path(image_path)
    checkpoint_path = Path(checkpoint_path)
    if not image_path.is_absolute():
        image_path = ROOT / image_path
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    if not image_path.is_file():
        raise ValueError("Image not found: " + str(image_path))
    if not checkpoint_path.is_file():
        raise ValueError("Checkpoint not found: " + str(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    model = build_model(checkpoint["backbone"], len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    from PIL import Image

    with Image.open(image_path) as image:
        tensor = make_transform(False)(image.convert("RGB"))
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    count = min(5, len(class_names))
    scores, indices = torch.topk(probabilities, count)
    for score, index in zip(scores.tolist(), indices.tolist()):
        print(class_names[index] + "\t" + format(score, ".4f"))
