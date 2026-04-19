import torch
import torch.nn as nn
import torch.nn.init as init
import math


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        reduced_planes = max(1, in_planes // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Linear(in_planes, reduced_planes, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_planes, in_planes, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)

        avg_out = self.shared_mlp(avg_out)
        max_out = self.shared_mlp(max_out)

        channel_attn = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * channel_attn


class TextGuidedSpatialAttention(nn.Module):
    """
    Text-guided spatial attention:
    - text_feat acts as Query
    - image spatial features act as Key/Value
    """
    def __init__(self, in_channels, text_dim, attn_dim=256):
        super(TextGuidedSpatialAttention, self).__init__()

        self.in_channels = in_channels
        self.text_dim = text_dim
        self.attn_dim = attn_dim

        # project text embedding -> query
        self.query_proj = nn.Linear(text_dim, attn_dim, bias=False)

        # project image feature map -> key/value
        self.key_proj = nn.Conv2d(in_channels, attn_dim, kernel_size=1, bias=False)
        self.value_proj = nn.Conv2d(in_channels, attn_dim, kernel_size=1, bias=False)

        # convert attended feature to spatial map
        self.out_proj = nn.Conv2d(attn_dim, 1, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)

    def forward(self, x, text_feat):
        """
        x: [B, C, H, W]
        text_feat: [B, D]
        return:
            x_attended: [B, C, H, W]
            spatial_attn_map: [B, 1, H, W]
        """
        b, c, h, w = x.shape
        n = h * w

        # Query from text
        q = self.query_proj(text_feat)  # [B, A]
        q = q.unsqueeze(1)              # [B, 1, A]

        # Key/Value from image
        k = self.key_proj(x)            # [B, A, H, W]
        v = self.value_proj(x)          # [B, A, H, W]

        k = k.flatten(2).transpose(1, 2)   # [B, N, A]
        v = v.flatten(2).transpose(1, 2)   # [B, N, A]

        # Cross-attention scores: Q x K^T
        attn_scores = torch.bmm(q, k.transpose(1, 2))  # [B, 1, N]
        attn_scores = attn_scores / math.sqrt(self.attn_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, 1, N]

        # Weighted sum over V
        attended = torch.bmm(attn_weights, v)  # [B, 1, A]
        attended = attended.transpose(1, 2).unsqueeze(-1)  # [B, A, 1, 1]

        # Broadcast attended text-guided context to spatial size
        attended_map = attended.expand(-1, -1, h, w)  # [B, A, H, W]

        # Produce spatial attention map
        spatial_attn = self.sigmoid(self.out_proj(attended_map))  # [B, 1, H, W]

        return x * spatial_attn, spatial_attn


class TextGuidedCBAM(nn.Module):
    def __init__(self, in_channels, text_dim, reduction=16, attn_dim=256):
        super(TextGuidedCBAM, self).__init__()

        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = TextGuidedSpatialAttention(
            in_channels=in_channels,
            text_dim=text_dim,
            attn_dim=attn_dim
        )

    def forward(self, x, text_feat):
        """
        x: [B, C, H, W]
        text_feat: [B, D]
        """
        x = self.channel_attention(x)
        x, spatial_attn = self.spatial_attention(x, text_feat)
        return x, spatial_attn