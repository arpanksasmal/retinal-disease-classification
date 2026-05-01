"""
RetinalCNN - EfficientNet-B3 Pretrained Backbone
v4: Unfreeze ALL layers for proper fine-tuning on small dataset
"""

import torch
import torch.nn as nn
from torchvision import models


class RetinalCNN(nn.Module):
    def __init__(self, num_classes: int = 5, dropout: float = 0.4):
        super().__init__()

        self.backbone = self._load_backbone()

        # v4 FIX: Unfreeze ALL layers
        # Previously freezing too many layers = barely anything learning
        for param in self.backbone.parameters():
            param.requires_grad = True

        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features  # 1536 for B3
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def _load_backbone(self):
        try:
            backbone = models.efficientnet_b3(
                weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
            )
            print("EfficientNet-B3 loaded with IMAGENET1K_V1 weights.")
            return backbone
        except Exception as e1:
            print(f"New weights API failed: {e1}")

        try:
            backbone = models.efficientnet_b3(pretrained=True)
            print("EfficientNet-B3 loaded with pretrained=True.")
            return backbone
        except Exception as e2:
            print(f"Legacy API also failed: {e2}")
            print("WARNING: Loading without pretrained weights.")
            return models.efficientnet_b3(weights=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


if __name__ == "__main__":
    model = RetinalCNN(num_classes=5)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape     : {out.shape}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params : {trainable:,} / {total:,}")
