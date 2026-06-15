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

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        # Truy vấn (Query - Q) từ đặc trưng hình ảnh
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        
        # Khóa (Key - K) từ đặc trưng văn bản
        self.k_proj = nn.Linear(dim, dim)
        
        # Giá trị (Value - V) từ đặc trưng văn bản (Thay đổi cốt lõi!)
        self.v_proj = nn.Linear(dim, dim)
        
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, img_feat, txt_feat, attention_mask=None):
        b, c, h, w = img_feat.shape
        _, l, _ = txt_feat.shape
        
        # 1. Q từ Image: [B, heads, H*W, C/heads]
        q = self.q_proj(img_feat).view(b, self.num_heads, c // self.num_heads, h * w).transpose(-2, -1)
        
        # 2. K từ Text: [B, heads, L, C/heads]
        k = self.k_proj(txt_feat).view(b, l, self.num_heads, c // self.num_heads).transpose(1, 2)
        
        # 3. V từ Text: [B, heads, L, C/heads]
        v = self.v_proj(txt_feat).view(b, l, self.num_heads, c // self.num_heads).transpose(1, 2)
        
        # 4. Tính toán Ma trận Attention: q @ k^T -> [B, heads, H*W, L]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Loại bỏ nhiễu từ các token <pad>
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2) # -> [B, 1, 1, L]
            attn = attn.masked_fill(mask == 0, -1e9)

        # 5. Trọng số Attention (Softmax) -> Dung hợp trọn vẹn ngữ cảnh ngôn ngữ
        attn_weights = F.softmax(attn, dim=-1) # -> [B, heads, H*W, L]
        
        # 6. Weighted Sum (Text Context) đắp vào Image -> [B, heads, H*W, C/heads]
        text_context = attn_weights @ v
        
        # Phục hồi kích thước về ảnh 2D ban đầu
        out = text_context.transpose(-2, -1).reshape(b, c, h, w)
        return self.out_proj(out)

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
        self.cross_attn = CrossAttention(dim, num_heads)
        self.restormer = RestormerBlock(dim, num_heads)

    def forward(self, img_feat, txt_feat, attention_mask=None):
        # 1. Text hướng dẫn Image (Cross Attention)
        guided_img = img_feat + self.cross_attn(img_feat, txt_feat, attention_mask)
        # 2. Tinh chỉnh đặc trưng bằng Restormer (Self Attention)
        refined_img = self.restormer(guided_img)
        return refined_img, txt_feat