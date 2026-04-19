import torch
import torch.nn as nn
import torch.nn.init as init

# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        reduced_planes = max(1, in_planes // reduction)  # Ensure at least 1 channel
        
        # Adaptive pooling to reduce spatial dimensions to 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Output shape: (B, C, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP for channel-wise attention
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_planes, reduced_planes, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_planes, in_planes, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input tensor of shape (batch_size, channels, height, width).
        # Output tensor with channel attention applied, same shape as input.
        b, c, _, _ = x.size()
        
        # Compute channel-wise avg and max
        avg_out = self.avg_pool(x).view(b, c)  # Shape: (batch_size, channels)
        max_out = self.max_pool(x).view(b, c)  # Shape: (batch_size, channels)

        # Pass through shared MLP
        avg_out = self.shared_mlp(avg_out)  # Shape: (batch_size, channels)
        max_out = self.shared_mlp(max_out)  # Shape: (batch_size, channels)

        # Combine and apply sigmoid to get attention weights
        channel_attn = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)  # Shape: (batch_size, channels, 1, 1)
        
        return x * channel_attn



# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        # Convolutional layer to generate spatial attention map
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)   # Shape: (batch_size, 1, height, width)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Shape: (batch_size, 1, height, width)
        
        spatial_feat = torch.cat([avg_out, max_out], dim=1) # Shape: (batch_size, 2, height, width)
        
        spatial_attn = self.sigmoid(self.conv(spatial_feat)) # Shape: (batch_size, 1, height, width)
        return x * spatial_attn

# CBAM Module
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        self._init_weights()
        
    #init weight for the CBAM module
    # - Conv2d: Kaiming normal initialization with fan_out mode
    # - Linear: Normal initialization with std=0.001, bias to 0
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass for CBAM.
        Input tensor of shape (batch_size, channels, height, width)
        Output tensor with channel and spatial attention applied, same shape as input.
        """
        # Apply channel attention
        x = self.channel_attention(x)
        # Apply spatial attention
        x = self.spatial_attention(x)
        return x


