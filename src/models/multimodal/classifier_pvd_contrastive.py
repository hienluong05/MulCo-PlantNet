from typing import List

import torch
import torch.nn as nn

from src.models.backbones.text.clip_text_encoder import CLIPTextEncoder
from src.models.backbones.vision.convnext_text_guided_cbam_encoder import ConvNeXtTextGuidedCBAMEncoder
from src.models.multimodal.classifier_pvd_base import MultimodalPVDClassifier


class MultimodalTextGuidedPVDInfoNCESupCon(nn.Module):
    """
    Version 3:
    - Text-guided CBAM image encoder
    - Pre-extracted text features
    - PVD fusion
    - support pre-fusion contrastive + post-fusion supervised contrastive
    """
    def __init__(
        self,
        num_classes: int = 28,
        text_input_dim: int = 768,
        image_input_dim: int = 1024,
        proj_dim: int = 512,
        proj_hidden_dim: int = 768,
        pvd_hidden_dim: int = 768,
        cls_hidden_dim: int = 1024,
        dropout: float = 0.3,
        normalize_projection: bool = True,
        device: str = "cpu"
    ):
        super().__init__()
        self.device = device

        self.image_encoder = ConvNeXtTextGuidedCBAMEncoder(
            text_dim=text_input_dim
        ).to(device)

        self.fusion_classifier = MultimodalPVDClassifier(
            image_input_dim=image_input_dim,
            text_input_dim=text_input_dim,
            proj_dim=proj_dim,
            proj_hidden_dim=proj_hidden_dim,
            pvd_hidden_dim=pvd_hidden_dim,
            cls_hidden_dim=cls_hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            normalize_projection=normalize_projection
        ).to(device)

    def forward(self, images: torch.Tensor, text_feat: torch.Tensor, return_attn: bool = False):
        # text_feat được giả định đã trích xuất sẵn và truyền vào dạng tensor [B, 768]

        if return_attn:
            image_feat, attn_maps = self.image_encoder(images, text_feat, return_attn=True)
        else:
            image_feat = self.image_encoder(images, text_feat, return_attn=False)
            attn_maps = None

        fusion_outputs = self.fusion_classifier(image_feat, text_feat, return_all=True)

        outputs = {
            "text_feat": text_feat,
            "image_feat": image_feat,
            "image_proj": fusion_outputs["image_proj"],
            "text_proj": fusion_outputs["text_proj"],
            "pvd_output": fusion_outputs["pvd_output"],
            "fused_feature": fusion_outputs["fused_feature"],
            "logits": fusion_outputs["logits"],
        }

        if attn_maps is not None:
            outputs["attn_maps"] = attn_maps

        return outputs