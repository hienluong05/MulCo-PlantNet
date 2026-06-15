import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1x1Classifier(nn.Module):
    """
    Kiến trúc FCN Classifier gốc (Dùng Conv2d 1x1 và GAP)
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.dropout = nn.Dropout2d(p=0.3)
        self.conv_1x1 = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.bn = nn.BatchNorm2d(num_classes)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv_1x1(x)
        x = self.bn(x)
        x = self.gap(x)
        return x.flatten(start_dim=1)

class GeMPool(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super(GeMPool, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

class MLPClassifier(nn.Module):
    def __init__(self, in_channels=512, hidden_dim=256, num_classes=28, dropout_rate=0.2):
        super().__init__()
        self.pool = GeMPool(p=3.0)
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return self.net(x)