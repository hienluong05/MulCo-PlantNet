import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from models.backbones.vision.convnext_cbam import ConvNeXt_CBAM


device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 1. CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Thư mục ảnh
DATA_ROOT = PROJECT_ROOT / "data" / "AIDG" / "dataset_PlantDoc" / "images" / "train"

# Checkpoint model đã train
CKPT_PATH = PROJECT_ROOT / "archive" / "best_model.pth" 

print("CKPT exists:", CKPT_PATH.exists())
print("CKPT size (MB):", CKPT_PATH.stat().st_size / (1024 * 1024))

# Output
OUTPUT_ROOT = PROJECT_ROOT / "data" / "features" # thư mục lưu features và metadata, sẽ được tạo nếu chưa tồn tại
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

USE_DEPTH_SUPPRESSED = False   # True: lấy ảnh *_depth_suppressed ; False: lấy ảnh gốc
OUTPUT_PREFIX = "train_depth" if USE_DEPTH_SUPPRESSED else "train_original"

FEATURE_PATH = OUTPUT_ROOT / f"{OUTPUT_PREFIX}_image_features_convnext_cbam.npy"
METADATA_PATH = OUTPUT_ROOT / f"{OUTPUT_PREFIX}_image_features_convnext_cbam_metadata.json"

NUM_CLASSES = 28   # sửa theo số lớp thực tế của bạn
BATCH_SIZE = 16
IMG_SIZE = 224
NUM_WORKERS = 2

# =========================
# 2. HELPERS
# =========================
def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in [".jpg", ".jpeg", ".png"]

def is_depth_image(path: Path) -> bool:
    return is_image_file(path) and "_depth_suppressed" in path.stem

def is_original_image(path: Path) -> bool:
    return is_image_file(path) and "_depth_suppressed" not in path.stem

def collect_images(data_root: Path, use_depth_suppressed: bool = True) -> List[Path]:
    all_paths = sorted([p for p in data_root.rglob("*") if is_image_file(p)])
    if use_depth_suppressed:
        return [p for p in all_paths if is_depth_image(p)]
    else:
        return [p for p in all_paths if is_original_image(p)]

def build_class_to_idx(image_paths: List[Path]) -> Dict[str, int]:
    class_names = sorted(list({p.parent.name for p in image_paths}))
    return {cls_name: i for i, cls_name in enumerate(class_names)}

# =========================
# 3. DATASET
# =========================
class PlantImageDataset(Dataset):
    def __init__(self, image_paths: List[Path], class_to_idx: Dict[str, int], transform=None):
        self.image_paths = image_paths
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        class_name = image_path.parent.name
        label = self.class_to_idx[class_name]

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "image_path": str(image_path.as_posix()),
            "class_name": class_name,
            "label": label,
        }

# =========================
# 4. LOAD MODEL
# =========================
def load_model(ckpt_path: Path, num_classes: int):
    model = ConvNeXt_CBAM(num_classes=num_classes)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # xử lý nếu checkpoint lưu từ DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device).eval()
    return model

# =========================
# 5. MAIN
# =========================
def main():
    print("DATA_ROOT:", DATA_ROOT)
    print("CKPT_PATH:", CKPT_PATH)
    print("FEATURE_PATH:", FEATURE_PATH)
    print("METADATA_PATH:", METADATA_PATH)
    print("USE_DEPTH_SUPPRESSED:", USE_DEPTH_SUPPRESSED)

    image_paths = collect_images(DATA_ROOT, use_depth_suppressed=USE_DEPTH_SUPPRESSED)
    print(f"Found {len(image_paths)} images.")

    if len(image_paths) == 0:
        raise ValueError("No images found.")

    class_to_idx = build_class_to_idx(image_paths)
    print("Num classes from folders:", len(class_to_idx))
    print("class_to_idx:", class_to_idx)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = PlantImageDataset(image_paths, class_to_idx, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = load_model(CKPT_PATH, NUM_CLASSES)

    all_features = []
    metadata = []

    with torch.no_grad():
        running_index = 0
        for batch in tqdm(loader, desc="Extracting image features"):
            images = batch["image"].to(device)

            feats = model.forward_features(images)   # [B, 1024]
            feats = feats.cpu().numpy()
            all_features.append(feats)

            batch_size_actual = feats.shape[0]
            for i in range(batch_size_actual):
                metadata.append({
                    "index": running_index,
                    "image_path": batch["image_path"][i],
                    "class_name": batch["class_name"][i],
                    "label": int(batch["label"][i]),
                    "image_type": "depth_suppressed" if USE_DEPTH_SUPPRESSED else "original"
                })
                running_index += 1

    all_features = np.concatenate(all_features, axis=0)

    np.save(FEATURE_PATH, all_features)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Saved features to:", FEATURE_PATH.resolve())
    print("Saved metadata to:", METADATA_PATH.resolve())
    print("Feature shape:", all_features.shape)
    print("Metadata length:", len(metadata))

if __name__ == "__main__":
    main()