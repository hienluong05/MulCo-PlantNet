from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from src.models.backbones.vision.convnext_cbam import ConvNeXt_CBAM


class ConvNeXtCBAMImageEncoder(nn.Module):
    def __init__(
        self,
        ckpt_path: Union[str, Path],
        num_classes: int = 28,
        device: str = "cpu"
    ):
        super().__init__()
        self.device = device
        self.model = ConvNeXt_CBAM(num_classes=num_classes)

        ckpt_path = Path(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict, strict=False)
        self.model = self.model.to(device).eval()

        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        feats = self.model.forward_features(images)   # [B, 1024]
        return feats