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


class PVDModule(nn.Module):
    """
    PVD nhận đầu vào là [V'; T'; V'*T'] có shape [B, 3*proj_dim]
    và xuất ra PVD_output có shape [B, proj_dim]
    """
    def __init__(
        self,
        proj_dim: int = 512,
        hidden_dim: int = 768,
        dropout: float = 0.2
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(proj_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, image_proj, text_proj):
        cross_interact = image_proj * text_proj                    # Bilinear interaction
        fused_input = torch.cat([image_proj, text_proj, cross_interact], dim=-1)   # [B, 3*proj_dim]
        pvd_output = self.mlp(fused_input)                         # [B, proj_dim]
        return pvd_output


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

        # 2. PVD module
        self.pvd = PVDModule(
            proj_dim=proj_dim,
            hidden_dim=pvd_hidden_dim,
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