from pathlib import Path
from typing import List, Union

import torch
import torch.nn as nn

from src.models.multimodal.classifier_pvd_contrastive import (
    MultimodalTextGuidedPVDInfoNCESupCon,
)


class MultiModalPipelineTextGuidedInfoNCESupCon(nn.Module):
    def __init__(
        self,
        ckpt_path: Union[str, Path],
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
        normalize_projection: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device

        self.model = MultimodalTextGuidedPVDInfoNCESupCon(
            num_classes=num_classes,
            clip_model_name=clip_model_name,
            clip_pretrained=clip_pretrained,
            text_input_dim=text_input_dim,
            image_input_dim=image_input_dim,
            proj_dim=proj_dim,
            proj_hidden_dim=proj_hidden_dim,
            pvd_hidden_dim=pvd_hidden_dim,
            cls_hidden_dim=cls_hidden_dim,
            dropout=dropout,
            normalize_projection=normalize_projection,
            device=device,
        ).to(device)

        ckpt_path = Path(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor, texts: List[str]) -> torch.Tensor:
        images = images.to(self.device)
        outputs = self.model(images, texts, return_attn=False)
        logits = outputs["logits"]
        return logits

    @torch.no_grad()
    def predict(self, images: torch.Tensor, texts: List[str]) -> torch.Tensor:
        logits = self.forward(images, texts)
        return torch.argmax(logits, dim=1)