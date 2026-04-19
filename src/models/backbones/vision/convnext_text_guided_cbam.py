import torch
import torch.nn as nn
import timm

from models.backbones.vision.text_guided_cbam import TextGuidedCBAM


class ConvNeXt_TextGuidedCBAM(nn.Module):
    def __init__(self, num_classes, text_dim=768):
        super(ConvNeXt_TextGuidedCBAM, self).__init__()

        self.model = timm.create_model("convnext_base", pretrained=True)

        self.tgcbam1 = TextGuidedCBAM(in_channels=128, text_dim=text_dim, attn_dim=128)
        self.tgcbam2 = TextGuidedCBAM(in_channels=256, text_dim=text_dim, attn_dim=256)
        self.tgcbam3 = TextGuidedCBAM(in_channels=512, text_dim=text_dim, attn_dim=256)
        self.tgcbam4 = TextGuidedCBAM(in_channels=1024, text_dim=text_dim, attn_dim=512)

        self.norm4 = nn.BatchNorm2d(1024)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        in_features = self.model.num_features
        self.model.head = nn.Linear(in_features, num_classes)

    def forward(self, x, text_feat):
        """
        x: [B, 3, H, W]
        text_feat: [B, text_dim]
        """
        attn_maps = {}

        x = self.model.stem(x)

        x = self.model.stages[0](x)
        x_attn, attn_maps["stage1"] = self.tgcbam1(x, text_feat)
        x = x + x_attn

        x = self.model.stages[1](x)
        x_attn, attn_maps["stage2"] = self.tgcbam2(x, text_feat)
        x = x + x_attn

        x = self.model.stages[2](x)
        x_attn, attn_maps["stage3"] = self.tgcbam3(x, text_feat)
        x = x + x_attn

        x = self.model.stages[3](x)
        x_attn, attn_maps["stage4"] = self.tgcbam4(x, text_feat)
        x = x + x_attn

        x = self.norm4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        logits = self.model.head(x)

        return logits, attn_maps