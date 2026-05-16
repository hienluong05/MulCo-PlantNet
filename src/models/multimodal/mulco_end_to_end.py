import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from src.models.backbones.vision.convnext_cbam import ConvNeXt_CBAM
from src.models.fusion.mulco_fusion import MulCoFusionBlock
from src.models.multimodal.mulco_classifier import Conv1x1Classifier

class MulCoEndToEnd(nn.Module):
    def __init__(self, num_classes=28, proj_dim=512):
        super().__init__()
        self.image_backbone = ConvNeXt_CBAM(num_classes=num_classes)
        self.text_backbone = AutoModel.from_pretrained("roberta-base")
        
        self.img_proj = nn.Conv2d(1024, proj_dim, kernel_size=1)
        self.txt_proj = nn.Linear(768, proj_dim)
        
        self.fusion_blocks = nn.ModuleList([
            # Sử dụng 1 khối Fusion duy nhất để chống Overfitting tối đa
            MulCoFusionBlock(dim=proj_dim, num_heads=8) for _ in range(1)
        ])
        
        self.classifier = Conv1x1Classifier(in_channels=proj_dim, num_classes=num_classes)

    def forward(self, images, input_ids, attention_mask):
        img_feat = self.image_backbone.forward_features_spatial(images) 
        
        txt_out = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = txt_out.last_hidden_state
        
        img_feat = self.img_proj(img_feat)
        txt_feat = self.txt_proj(txt_feat)
        
        for block in self.fusion_blocks:
            img_feat, txt_feat = block(img_feat, txt_feat)
            
        logits = self.classifier(img_feat)
        return logits