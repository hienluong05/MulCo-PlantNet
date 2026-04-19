import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 1024,
        num_classes: int = 28,
        dropout: float = 0.3
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   # 512 -> 1024
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)  # 1024 -> num_classes
        )

    def forward(self, fused_feature):
        logits = self.classifier(fused_feature)
        return logits