import torch
import torch.nn as nn

class Conv1x1Classifier(nn.Module):
    """Khối phân loại sử dụng Conv 1x1 giảm chiều -> Flatten -> FC duy nhất"""
    def __init__(self, in_channels, num_classes, spatial_size=(7, 7)):
        super().__init__()
        # Giảm chiều kênh nhưng giữ nhận thức không gian (Spatial awareness)
        self.conv_1x1 = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.act = nn.GELU()
        
        # Flatten và FC duy nhất
        flatten_dim = num_classes * spatial_size[0] * spatial_size[1]
        self.fc = nn.Linear(flatten_dim, num_classes)

    def forward(self, x):
        x = self.conv_1x1(x)       # -> [B, num_classes, H, W]
        x = self.act(x)
        x = x.flatten(start_dim=1) # -> [B, num_classes * H * W]
        x = self.fc(x)             # -> [B, num_classes]
        return x