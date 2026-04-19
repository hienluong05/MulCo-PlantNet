import json
import numpy as np
import torch
from torch.utils.data import Dataset


class MultiModalFeatureDataset(Dataset):
    def __init__(self, image_feature_path, text_feature_path, metadata_path):
        self.image_features = np.load(image_feature_path)
        self.text_features = np.load(text_feature_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        assert len(self.image_features) == len(self.text_features) == len(self.metadata), \
            "Image features, text features, and metadata must have the same length."

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]

        image_feat = torch.tensor(self.image_features[idx], dtype=torch.float32)
        text_feat = torch.tensor(self.text_features[idx], dtype=torch.float32)
        label = torch.tensor(item["label"], dtype=torch.long)

        return {
            "image_feat": image_feat,
            "text_feat": text_feat,
            "label": label,
            "image_name": item.get("image_name", ""),
            "class_name": item.get("class_name", "")
        }