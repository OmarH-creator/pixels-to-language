import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    Standard ResNet Basic Block (He et al., 2015).
    Contains:
      - Two 3x3 Conv2d layers with BatchNorm and ReLU.
      - A shortcut connection: pure identity when in_c == out_c and stride == 1,
        or a 1x1 Conv + BatchNorm only when spatial size or channel count changes.
    """
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        # Conv 1 (downsamples spatially if stride > 1)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.relu  = nn.ReLU(inplace=True)

        # Conv 2 (always stride 1)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)

        # Shortcut path
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.relu(out + identity)
        return out


class BaselineCNN_V3_ResNet18(nn.Module):
    """
    ResNet-18 Style Architecture adapted for from-scratch bird classification.
    Total depth: 1 stem conv + 8 BasicBlocks (16 convs) + 1 Linear head = 18 weighted layers.
    Parameters: ~2,845,416 (lighter and deeper than V2).
    """
    def __init__(self, num_classes: int = 200, in_channels: int = 3):
        super().__init__()

        # 1. Stem: Quick initial downsampling (224x224 -> 56x56)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),  # -> 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                            # -> 56x56
        )

        # 2. ResNet Stages (2 blocks per stage)
        # Stage 1: 56x56, 32 channels (no spatial downsampling)
        self.stage1 = nn.Sequential(
            BasicBlock(32, 32, stride=1),
            BasicBlock(32, 32, stride=1)
        )

        # Stage 2: 28x28, 64 channels (downsampled by stride=2 on first block)
        self.stage2 = nn.Sequential(
            BasicBlock(32, 64, stride=2),
            BasicBlock(64, 64, stride=1)
        )

        # Stage 3: 14x14, 128 channels (downsampled by stride=2 on first block)
        self.stage3 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128, stride=1)
        )

        # Stage 4: 7x7, 256 channels (downsampled by stride=2 on first block)
        self.stage4 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256, stride=1)
        )

        # 3. Global Pooling & Classifier Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),                # Slightly stronger dropout to fight overfitting
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    # Quick sanity check
    model = BaselineCNN_V3_ResNet18(num_classes=200)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"BaselineCNN_V3_ResNet18 Output shape: {out.shape}")
    print(f"Total trainable parameters: {params:,}")
