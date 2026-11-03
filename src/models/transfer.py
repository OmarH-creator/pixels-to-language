# ImageNet ResNet with a new species classification layer.
from torch import nn
from torchvision import models

BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights),
    "resnet34": (models.resnet34, models.ResNet34_Weights),
    "resnet50": (models.resnet50, models.ResNet50_Weights),
}


def build_model(backbone, num_classes, pretrained):
    if backbone not in BACKBONES:
        choices = ", ".join(BACKBONES)
        raise ValueError("Unknown backbone '" + backbone + "'. Choose " + choices + ".")
    builder, weights = BACKBONES[backbone]
    model = builder(weights=weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
