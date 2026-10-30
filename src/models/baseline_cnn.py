import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Two 3x3 Convolutions + BatchNorm + ReLU + Residual Skip Connection,
    followed by MaxPool2d(2, 2) to halve spatial dimensions.
    """
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)

        # 1x1 shortcut so in_c channels match out_c channels when we add them
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.relu(out + identity)  # Residual skip connection!
        out = self.pool(out)
        return out


class BaselineCNN(nn.Module):
    """
    Upgraded 5-Stage Residual CNN from scratch for 200-species bird classification.
    """
    def __init__(self, num_classes: int = 200, in_channels: int = 3):
        super().__init__()

        # 1. Feature Extractor (5 Residual Stages)
        # Input shape: (Batch, 3, 224, 224)
        self.stage1 = ResidualBlock(in_channels, 32)   # -> (Batch, 32, 112, 112)
        self.stage2 = ResidualBlock(32, 64)            # -> (Batch, 64, 56, 56)
        self.stage3 = ResidualBlock(64, 128)           # -> (Batch, 128, 28, 28)
        self.stage4 = ResidualBlock(128, 256)          # -> (Batch, 256, 14, 14)
        self.stage5 = ResidualBlock(256, 512)          # -> (Batch, 512, 7, 7)

        # 2. Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # -> (Batch, 512, 1, 1)

        # 3. Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),                     # -> (Batch, 512)
            nn.Dropout(p=0.3),                # Regularization
            nn.Linear(512, num_classes),      # -> (Batch, 200)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits