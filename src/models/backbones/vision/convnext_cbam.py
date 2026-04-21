import torch
import torch.nn as nn
import timm

from src.models.backbones.vision.cbam import CBAM


class ConvNeXt_CBAM(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeXt_CBAM, self).__init__()

        self.model = timm.create_model("convnext_base", pretrained=True)

        self.cbam1 = CBAM(in_channels=128)
        self.cbam2 = CBAM(in_channels=256)
        self.cbam3 = CBAM(in_channels=512)
        self.cbam4 = CBAM(in_channels=1024)

        self.norm4 = nn.BatchNorm2d(1024)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        in_features = self.model.num_features
        self.model.head = nn.Linear(in_features, num_classes)

    def forward_features(self, x):
        x = self.model.stem(x)

        x = self.model.stages[0](x)
        x = x + self.cbam1(x)

        x = self.model.stages[1](x)
        x = x + self.cbam2(x)

        x = self.model.stages[2](x)
        x = x + self.cbam3(x)

        x = self.model.stages[3](x)
        x = x + self.cbam4(x)
        x = self.norm4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x, return_features=False):
        feats = self.forward_features(x)
        logits = self.model.head(feats)

        if return_features:
            return logits, feats
        return logits