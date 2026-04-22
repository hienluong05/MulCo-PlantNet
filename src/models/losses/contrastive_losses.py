import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageTextInfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE loss for image-text alignment.
    Input:
        image_proj: [B, D]
        text_proj: [B, D]
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_proj: torch.Tensor, text_proj: torch.Tensor) -> torch.Tensor:
        image_proj = F.normalize(image_proj, dim=-1)
        text_proj = F.normalize(text_proj, dim=-1)

        logits = image_proj @ text_proj.t() / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)

        return 0.5 * (loss_i2t + loss_t2i)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    Reference idea: Supervised Contrastive Learning.
    Input:
        features: [B, D]
        labels: [B]
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        features = F.normalize(features, dim=-1)

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)

        if labels.shape[0] != batch_size:
            raise ValueError("labels size does not match features batch size")

        mask = torch.eq(labels, labels.T).float().to(device)

        logits = torch.matmul(features, features.T) / self.temperature

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positive_counts = mask.sum(dim=1)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (positive_counts + 1e-12)

        valid_mask = (positive_counts > 0).float()
        loss = -mean_log_prob_pos * valid_mask

        return loss.sum() / (valid_mask.sum() + 1e-12)