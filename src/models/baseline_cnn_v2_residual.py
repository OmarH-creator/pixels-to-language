import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Two 3x3 Convolutions + BatchNorm + ReLU with Residual Skip Connection,
    followed by MaxPool2d(2, 2) to halve spatial dimensions.
    """
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)

        # 1x1 conv shortcut to match channels when adding identity
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.relu(out + identity)  # Residual skip connection
        out = self.pool(out)
        return out


class BaselineCNN_V2_Residual(nn.Module):
    """
    V2 Upgraded Baseline CNN (5 Stages, Double Conv + Residual Connections per stage).
    Parameters: ~4,714,640
    Feature Map Size before pooling: 7x7 with 512 channels.
    """
    def __init__(self, num_classes: int = 200, in_channels: int = 3):
        super().__init__()

        # 5 Residual Stages
        self.stage1 = ResidualBlock(in_channels, 32)   # -> [B, 32, 112, 112]
        self.stage2 = ResidualBlock(32, 64)            # -> [B, 64, 56, 56]
        self.stage3 = ResidualBlock(64, 128)           # -> [B, 128, 28, 28]
        self.stage4 = ResidualBlock(128, 256)          # -> [B, 256, 14, 14]
        self.stage5 = ResidualBlock(256, 512)          # -> [B, 512, 7, 7]

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
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
