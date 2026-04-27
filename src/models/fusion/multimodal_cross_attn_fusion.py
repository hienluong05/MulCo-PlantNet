import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 768, output_dim: int = 512, dropout: float = 0.1, normalize: bool = False):
        super().__init__()
        self.normalize = normalize
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.net(x)
        if self.normalize:
            x = F.normalize(x, dim=-1)
        return x


class TokenLevelCrossAttentionModule(nn.Module):
    """
    Sử dụng Cross-Attention (Text query Image) để tìm vùng bệnh dựa trên ngữ cảnh từ text.
    Hỗ trợ đầu vào dạng Global [B, D] (tự động unsqueeze) hoặc dạng Token [B, N, D].
    """
    def __init__(self, proj_dim: int = 512, num_heads: int = 8, hidden_dim: int = 768, dropout: float = 0.2):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(proj_dim)
        self.norm2 = nn.LayerNorm(proj_dim)
        self.ffn = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, image_proj, text_proj):
        if image_proj.dim() == 2:
            image_proj = image_proj.unsqueeze(1)
        if text_proj.dim() == 2:
            text_proj = text_proj.unsqueeze(1)
            
        attn_output, attn_weights = self.cross_attn(query=text_proj, key=image_proj, value=image_proj)
        
        out1 = self.norm1(text_proj + attn_output)
        out2 = self.norm2(out1 + self.ffn(out1))
        fused_output = out2.mean(dim=1) # [B, proj_dim]
        
        return fused_output, attn_weights


class MultimodalCrossAttnFusion(nn.Module):
    def __init__(self, image_input_dim=1024, text_input_dim=768, proj_dim=512, proj_hidden_dim=768, pvd_hidden_dim=768, dropout=0.2, normalize_projection=False):
        super().__init__()
        self.image_projection = ProjectionHead(input_dim=image_input_dim, hidden_dim=proj_hidden_dim, output_dim=proj_dim, dropout=dropout, normalize=normalize_projection)
        self.text_projection = ProjectionHead(input_dim=text_input_dim, hidden_dim=proj_hidden_dim, output_dim=proj_dim, dropout=dropout, normalize=normalize_projection)
        self.cross_attn = TokenLevelCrossAttentionModule(proj_dim=proj_dim, num_heads=8, hidden_dim=pvd_hidden_dim, dropout=dropout)
        self.gate = nn.Sequential(nn.Linear(proj_dim * 2, proj_dim), nn.Sigmoid())

    def forward(self, image_feat, text_feat, return_all=False):
        if image_feat.dim() == 4:
            B, C, H, W = image_feat.shape
            image_feat = image_feat.view(B, C, H * W).transpose(1, 2)

        image_proj = self.image_projection(image_feat)
        text_proj = self.text_projection(text_feat)

        pvd_output, attn_weights = self.cross_attn(image_proj, text_proj)

        pooled_image = image_proj.mean(dim=1) if image_proj.dim() == 3 else image_proj
        pooled_text = text_proj.mean(dim=1) if text_proj.dim() == 3 else text_proj

        gate_weight = self.gate(torch.cat([pooled_image, pooled_text], dim=-1))
        fused_feature = (gate_weight * pooled_image) + ((1 - gate_weight) * pooled_text) + pvd_output

        if return_all:
            return {"image_proj": image_proj, "text_proj": text_proj, "pvd_output": pvd_output, "fused_feature": fused_feature, "attn_weights": attn_weights}
        return fused_feature