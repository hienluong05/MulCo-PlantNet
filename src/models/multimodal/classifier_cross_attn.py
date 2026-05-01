import torch.nn as nn
from pathlib import Path
import sys

# Lấy đường dẫn thư mục gốc (lùi ra 3 cấp thư mục: multimodal -> models -> src -> MulCo-PlantNet)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))
from src.models.fusion.fusion_cross_attn import MultimodalCrossAttnFusion 

class MultimodalCrossAttnClassifier(nn.Module):
    def __init__(
        self, image_input_dim=1024, text_input_dim=768, proj_dim=512, 
        proj_hidden_dim=768, pvd_hidden_dim=768, cls_hidden_dim=1024, 
        num_classes=28, dropout=0.2, normalize_projection=False
    ):
        super().__init__()
        self.fusion = MultimodalCrossAttnFusion(
            image_input_dim=image_input_dim,
            text_input_dim=text_input_dim,
            proj_dim=proj_dim,
            proj_hidden_dim=proj_hidden_dim,
            pvd_hidden_dim=pvd_hidden_dim,
            dropout=dropout,
            normalize_projection=normalize_projection
        )
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, cls_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden_dim, num_classes)
        )

    def forward(self, image_feat, text_feat, return_all=False):
        fusion_output = self.fusion(image_feat, text_feat, return_all=return_all)
        if return_all:
            logits = self.classifier(fusion_output["fused_feature"])
            fusion_output["logits"] = logits
            return fusion_output
        return self.classifier(fusion_output)