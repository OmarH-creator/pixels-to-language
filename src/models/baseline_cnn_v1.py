import torch
import torch.nn as nn


class BaselineCNN_V1(nn.Module):
    """
    V1 Minimal Baseline CNN (4 Stages, single 3x3 conv per stage).
    Parameters: ~440,296
    Feature Map Size before pooling: 14x14 with 256 channels.
    """

    def __init__(self, num_classes: int = 200, in_channels: int = 3):
        super().__init__()

        def _make_block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        # 4 Convolutional stages
        self.stage1 = _make_block(in_channels, 32)   # -> [B, 32, 112, 112]
        self.stage2 = _make_block(32, 64)            # -> [B, 64, 56, 56]
        self.stage3 = _make_block(64, 128)           # -> [B, 128, 28, 28]
        self.stage4 = _make_block(128, 256)          # -> [B, 256, 14, 14]

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits
