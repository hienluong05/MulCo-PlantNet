from typing import List
import torch
import torch.nn as nn
import open_clip


class CLIPTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str = "cpu",
        normalize: bool = True
    ):
        super().__init__()
        self.device = device
        self.normalize = normalize
        self.model_name = model_name
        self.pretrained = pretrained

        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.device)
        feats = self.model.encode_text(tokens)

        if self.normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats