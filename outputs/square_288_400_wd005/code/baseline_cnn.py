import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Two 3x3 Convolutions + BatchNorm + ReLU + Residual Skip Connection,
    followed by MaxPool2d(2, 2) to halve spatial dimensions.
    """
    ##########################################################################
    # DEEPER MODEL: pool=False keeps the size, for the extra blocks in a stage
    def __init__(self, in_c: int, out_c: int, pool: bool = True):
    ##########################################################################
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)

        ##########################################################################
        # DEEPER MODEL: when channels already match, the shortcut passes x unchanged
        if in_c == out_c:
            self.shortcut = nn.Identity()
        else:
            # 1x1 shortcut so in_c channels match out_c channels when we add them
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_c),
            )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()
        ##########################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.relu(out + identity)  # Residual skip connection!
        out = self.pool(out)
        return out


##########################################################################
# DEEPER MODEL: one normal block (halves the size), then extra blocks at that size
def make_stage(in_c: int, out_c: int, blocks: int) -> nn.Sequential:
    layers = [ResidualBlock(in_c, out_c)]
    for _ in range(blocks - 1):
        layers.append(ResidualBlock(out_c, out_c, pool=False))
    return nn.Sequential(*layers)
##########################################################################


class BaselineCNN(nn.Module):
    """
    Upgraded 5-Stage Residual CNN from scratch for 200-species bird classification.
    """
    def __init__(self, num_classes: int = 200, in_channels: int = 3):
        super().__init__()

        # 1. Feature Extractor (5 Residual Stages)
        ##########################################################################
        # DEEPER MODEL + BIGGER IMAGES: 12 blocks x 2 convs + 1 Linear = 25 main layers
        # Input shape: (Batch, 3, 288, 288)
        self.stage1 = make_stage(in_channels, 32, blocks=2)   # -> (Batch, 32, 144, 144)
        self.stage2 = make_stage(32, 64, blocks=2)            # -> (Batch, 64, 72, 72)
        self.stage3 = make_stage(64, 128, blocks=3)           # -> (Batch, 128, 36, 36)
        self.stage4 = make_stage(128, 256, blocks=3)          # -> (Batch, 256, 18, 18)
        self.stage5 = make_stage(256, 512, blocks=2)          # -> (Batch, 512, 9, 9)
        ##########################################################################

        ##########################################################################
        # ATTENTION POOLING: scores each of the 9x9 positions (replaces global average pooling)
        self.attention = nn.Conv2d(512, 1, kernel_size=1)  # -> (Batch, 1, 9, 9)
        ##########################################################################

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

        ##########################################################################
        # ATTENTION POOLING: weighted average, so important positions count more
        weights = torch.softmax(self.attention(x).flatten(2), dim=2)  # -> (Batch, 1, 81)
        x = (x.flatten(2) * weights).sum(dim=2)                       # -> (Batch, 512)
        ##########################################################################
        logits = self.classifier(x)
        return logits