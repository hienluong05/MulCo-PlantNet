from pathlib import Path
from typing import List, Union

import torch
import torch.nn as nn

from src.models.backbones.text.clip_text_encoder import CLIPTextEncoder
from src.models.backbones.vision.convnext_cbam_image_encoder import ConvNeXtCBAMImageEncoder
from src.models.multimodal.classifier_pvd_base import MultimodalPVDClassifier


class MultiModalPipelineConvNeXtCBAMCLIPPVD(nn.Module):
    def __init__(
        self,
        image_ckpt_path: Union[str, Path],
        fusion_ckpt_path: Union[str, Path],
        num_classes: int = 28,
        clip_model_name: str = "ViT-L-14",
        clip_pretrained: str = "openai",
        image_input_dim: int = 1024,
        text_input_dim: int = 768,
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

        self.image_encoder = ConvNeXtCBAMImageEncoder(
            ckpt_path=image_ckpt_path,
            num_classes=num_classes,
            device=device
        )

        self.text_encoder = CLIPTextEncoder(
            model_name=clip_model_name,
            pretrained=clip_pretrained,
            device=device,
            normalize=True
        )

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

        fusion_ckpt_path = Path(fusion_ckpt_path)
        state_dict = torch.load(fusion_ckpt_path, map_location="cpu")
        self.fusion_classifier.load_state_dict(state_dict, strict=True)
        self.fusion_classifier = self.fusion_classifier.to(device).eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor, texts: List[str]) -> torch.Tensor:
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(texts)
        logits = self.fusion_classifier(image_features, text_features)
        return logits

    @torch.no_grad()
    def predict(self, images: torch.Tensor, texts: List[str]) -> torch.Tensor:
        logits = self.forward(images, texts)
        return torch.argmax(logits, dim=1)