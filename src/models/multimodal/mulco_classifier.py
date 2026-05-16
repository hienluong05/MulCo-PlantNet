import torch
import torch.nn as nn

class Conv1x1Classifier(nn.Module):
    """
    Kiến trúc FCN Classifier gốc (Dùng Conv2d 1x1 và GAP)
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv_1x1 = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.bn = nn.BatchNorm2d(num_classes)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv_1x1(x)
        x = self.bn(x)
        x = self.gap(x)
        return x.flatten(start_dim=1)