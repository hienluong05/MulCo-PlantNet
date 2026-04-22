from typing import List

import torch
import torch.nn as nn

from src.models.backbones.text.clip_text_encoder import CLIPTextEncoder
from src.models.backbones.vision.convnext_text_guided_cbam_encoder import ConvNeXtTextGuidedCBAMEncoder
from src.models.multimodal.multimodal_pvd_classifier import MultimodalPVDClassifier


class MultimodalTextGuidedPVDClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 28,
        clip_model_name: str = "ViT-L-14",
        clip_pretrained: str = "openai",
        text_input_dim: int = 768,
        image_input_dim: int = 1024,
        proj_dim: int = 512,
        proj_hidden_dim: int = 768,
        pvd_hidden_dim: int = 768,
        cls_hidden_dim: int = 1024,
        dropout: float = 0.3,
        normalize_projection: bool = False,
        device: str = "cpu"
    ):
        super().__init__()
        self.device = device

        self.text_encoder = CLIPTextEncoder(
            model_name=clip_model_name,
            pretrained=clip_pretrained,
            device=device,
            normalize=True
        )

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

    def forward(self, images: torch.Tensor, texts: List[str], return_all=False):
        text_feat = self.text_encoder(texts)                  # [B, 768]
        image_feat = self.image_encoder(images, text_feat)    # [B, 1024]

        outputs = self.fusion_classifier(image_feat, text_feat, return_all=return_all)
        return outputs