import torch
import torch.nn as nn
import torch.nn.functional as F

class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.view(b, self.num_heads, -1, h * w)
        k = k.view(b, self.num_heads, -1, h * w)
        v = v.view(b, self.num_heads, -1, h * w)
        
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).view(b, c, h, w)
        out = self.project_out(out)
        return out

class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network"""
    def __init__(self, dim, expansion_factor=2.66):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=1, padding=1, groups=hidden_dim * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class RestormerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MDTA(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = GDFN(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm1(x.view(b, c, -1).transpose(-2, -1)).transpose(-2, -1).view(b, c, h, w)
        x = x + self.attn(x_norm)
        
        x_norm = self.norm2(x.view(b, c, -1).transpose(-2, -1)).transpose(-2, -1).view(b, c, h, w)
        x = x + self.ffn(x_norm)
        return x

class MulCoFusionBlock(nn.Module):
    """Module Dung hợp bao gồm Reversed Cross-Attention (tạm đơn giản hóa) và Restormer"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        # Có thể tách riêng Cross-Attention nếu muốn, hiện tại tích hợp xử lý ở đây
        self.restormer = RestormerBlock(dim, num_heads)

    def forward(self, img_feat, txt_feat):
        # (Tạm giả định img_feat đã được Text dẫn dắt qua Cross Attention ở bước ngoài hoặc xử lý ghép)
        # Tinh chỉnh đặc trưng bằng Restormer
        refined_img = self.restormer(img_feat)
        return refined_img, txt_feat