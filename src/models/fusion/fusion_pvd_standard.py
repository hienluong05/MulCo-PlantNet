import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 768,
        output_dim: int = 512,
        dropout: float = 0.1,
        normalize: bool = False
    ):
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


class TransformerFusionModule(nn.Module):
    """
    Sử dụng Multi-Head Self-Attention để fusion Image và Text.
    Đầu vào: image_proj [B, proj_dim], text_proj [B, proj_dim]
    Đầu ra: fused_output [B, proj_dim]
    """
    def __init__(
        self,
        proj_dim: int = 512,
        num_heads: int = 8,
        hidden_dim: int = 768,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, image_proj, text_proj):
        # Stack thành sequence có 2 token: [B, 2, proj_dim]
        seq = torch.stack([image_proj, text_proj], dim=1)
        
        # Pass qua Transformer
        out_seq = self.transformer(seq) # [B, 2, proj_dim]
        
        # Average pooling 2 token để ra vector đại diện chung
        fused_output = out_seq.mean(dim=1) # [B, proj_dim]
        
        return fused_output


class MultimodalPVDFusion(nn.Module):
    def __init__(
        self,
        image_input_dim: int = 1024,
        text_input_dim: int = 768,
        proj_dim: int = 512,
        proj_hidden_dim: int = 768,
        pvd_hidden_dim: int = 768,
        dropout: float = 0.2,
        normalize_projection: bool = False
    ):
        super().__init__()

        # 1. Projection heads
        self.image_projection = ProjectionHead(
            input_dim=image_input_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_dim,
            dropout=dropout,
            normalize=normalize_projection
        )

        self.text_projection = ProjectionHead(
            input_dim=text_input_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_dim,
            dropout=dropout,
            normalize=normalize_projection
        )

        # 2. Transformer Fusion module (thay thế PVD MLP cũ)
        self.pvd = TransformerFusionModule(
            proj_dim=proj_dim,
            num_heads=8,
            hidden_dim=pvd_hidden_dim,
            num_layers=2,
            dropout=dropout
        )

        # 3. Gating Mechanism (Cơ chế cổng học trọng số động)
        self.gate = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.Sigmoid()
        )

    def forward(self, image_feat, text_feat, return_all=False):
        # Step 1: project to common space
        image_proj = self.image_projection(image_feat)   # [B, proj_dim]
        text_proj = self.text_projection(text_feat)      # [B, proj_dim]

        # Step 2: PVD
        pvd_output = self.pvd(image_proj, text_proj)     # [B, proj_dim]

        # Step 3: Gated Residual fusion
        # Tạo mặt nạ học trọng số dựa trên thông tin cả 2 modality
        gate_weight = self.gate(torch.cat([image_proj, text_proj], dim=-1))
        fused_feature = (gate_weight * image_proj) + ((1 - gate_weight) * text_proj) + pvd_output   # [B, proj_dim]

        if return_all:
            return {
                "image_proj": image_proj,
                "text_proj": text_proj,
                "pvd_output": pvd_output,
                "fused_feature": fused_feature
            }

        return fused_feature