import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    Baseline Convolutional Neural Network built from scratch for bird classification.
    
    Architecture summary:
      - 4 Convolutional stages (feature extraction)
      - Each stage: Conv2d(3x3) -> BatchNorm -> ReLU -> MaxPool2d(2x2)
      - Classifier head: Global Average Pooling -> Dropout -> Linear(num_classes)
    """

    def __init__(self, num_classes: int = 200, in_channels: int = 3):
        super().__init__()

        def _make_block(in_c: int, out_c: int) -> nn.Sequential:
            """Helper: Conv -> BatchNorm -> ReLU -> MaxPool"""
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        # 1. Feature Extractor (Encoder Backbone)
        # Input shape: (Batch, 3, 224, 224)
        self.stage1 = _make_block(in_channels, 32)   # -> (Batch, 32, 112, 112)
        self.stage2 = _make_block(32, 64)            # -> (Batch, 64, 56, 56)
        self.stage3 = _make_block(64, 128)           # -> (Batch, 128, 28, 28)
        self.stage4 = _make_block(128, 256)          # -> (Batch, 256, 14, 14)

        # 2. Global Average Pooling
        # ResNet trick: averages spatial 14x14 down to 1x1, saving millions of parameters
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # -> (Batch, 256, 1, 1)

        # 3. Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),                     # -> (Batch, 256)
            nn.Dropout(p=0.3),                # Randomly zero out 30% of features to fight overfitting
            nn.Linear(256, num_classes),      # -> (Batch, 200) raw scores (logits)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass data flow.
        x shape: (Batch, 3, 224, 224)
        returns: (Batch, num_classes)
        """
        # Pass through the 4 convolutional stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Pool and classify
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    # --- Quick Sanity Check ---
    print("Testing BaselineCNN architecture...")
    model = BaselineCNN(num_classes=200)

    # Create dummy batch: 2 fake RGB images of size 224x224
    dummy_images = torch.randn(2, 3, 224, 224)
    output = model(dummy_images)

    print(f"Input shape:  {dummy_images.shape}")
    print(f"Output shape: {output.shape} (2 birds, 200 species scores)")

    # Test Loss calculation (CrossEntropy)
    dummy_labels = torch.tensor([42, 189])  # 2 fake target bird class IDs
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, dummy_labels)
    print(f"Calculated loss on dummy data: {loss.item():.4f}")

    # Count total trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
