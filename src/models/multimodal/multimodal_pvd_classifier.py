import torch
import torch.nn as nn

from src.models.fusion.multimodal_pvd_fusion import MultimodalPVDFusion
from src.models.heads.classification_head import ClassificationHead


class MultimodalPVDClassifier(nn.Module):
    def __init__(
        self,
        image_input_dim: int = 1024,
        text_input_dim: int = 768,
        proj_dim: int = 512,
        proj_hidden_dim: int = 768,
        pvd_hidden_dim: int = 768,
        cls_hidden_dim: int = 1024,
        num_classes: int = 28,
        dropout: float = 0.3,
        normalize_projection: bool = False
    ):
        super().__init__()

        self.fusion = MultimodalPVDFusion(
            image_input_dim=image_input_dim,
            text_input_dim=text_input_dim,
            proj_dim=proj_dim,
            proj_hidden_dim=proj_hidden_dim,
            pvd_hidden_dim=pvd_hidden_dim,
            dropout=dropout,
            normalize_projection=normalize_projection
        )

        self.classification_head = ClassificationHead(
            input_dim=proj_dim,
            hidden_dim=cls_hidden_dim,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(self, image_feat, text_feat, return_all=False):
        fusion_outputs = self.fusion(image_feat, text_feat, return_all=True)
        fused_feature = fusion_outputs["fused_feature"]

        logits = self.classification_head(fused_feature)

        if return_all:
            fusion_outputs["logits"] = logits
            return fusion_outputs

        return logits