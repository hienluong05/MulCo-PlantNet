import torch
import torch.nn as nn

from src.models.losses.contrastive_losses import ImageTextInfoNCELoss, SupConLoss


class InfoNCESupConLoss(nn.Module):
    def __init__(
        self,
        ce_weight: float = 1.0,
        itc_weight: float = 0.2,
        supcon_weight: float = 0.2,
        itc_temperature: float = 0.07,
        supcon_temperature: float = 0.07
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.itc_weight = itc_weight
        self.supcon_weight = supcon_weight

        self.ce_loss = nn.CrossEntropyLoss()
        self.itc_loss = ImageTextInfoNCELoss(temperature=itc_temperature)
        self.supcon_loss = SupConLoss(temperature=supcon_temperature)

    def forward(self, outputs: dict, labels: torch.Tensor):
        logits = outputs["logits"]
        image_proj = outputs["image_proj"]
        text_proj = outputs["text_proj"]
        fused_feature = outputs["fused_feature"]

        loss_ce = self.ce_loss(logits, labels)
        loss_itc = self.itc_loss(image_proj, text_proj)
        loss_supcon = self.supcon_loss(fused_feature, labels)

        total_loss = (
            self.ce_weight * loss_ce
            + self.itc_weight * loss_itc
            + self.supcon_weight * loss_supcon
        )

        return {
            "loss": total_loss,
            "loss_ce": loss_ce,
            "loss_itc": loss_itc,
            "loss_supcon": loss_supcon,
        }