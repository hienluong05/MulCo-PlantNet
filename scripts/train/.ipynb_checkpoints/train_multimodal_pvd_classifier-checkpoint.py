from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.datasets.multimodal_feature_dataset import MultiModalFeatureDataset
from src.models.multimodal.multimodal_pvd_classifier import MultimodalPVDClassifier


device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# CONFIG
# =========================
IMAGE_FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "encoded_image" / "train_original_image_features_convnext_cbam.npy"
TEXT_FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "encoded_text" / "train_text_embeddings_clip_vitl14.npy"
METADATA_PATH = PROJECT_ROOT / "data" / "features" / "encoded_image" / "train_original_image_features_convnext_cbam_metadata.json"

NUM_CLASSES = 28
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4

# =========================
# DATASET
# =========================
dataset = MultiModalFeatureDataset(
    image_feature_path=IMAGE_FEATURE_PATH,
    text_feature_path=TEXT_FEATURE_PATH,
    metadata_path=METADATA_PATH
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =========================
# MODEL
# =========================
model = MultimodalPVDClassifier(
    image_input_dim=1024,
    text_input_dim=768,
    proj_dim=512,
    proj_hidden_dim=768,
    pvd_hidden_dim=768,
    cls_hidden_dim=1024,
    num_classes=NUM_CLASSES,
    dropout=0.3,
    normalize_projection=False
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =========================
# TRAIN LOOP
# =========================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        image_feat = batch["image_feat"].to(device)
        text_feat = batch["text_feat"].to(device)
        labels = batch["label"].to(device)

        logits = model(image_feat, text_feat)   # [B, num_classes]
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = total_loss / total
    epoch_acc = correct / total

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")