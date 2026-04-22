import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset


def normalize_caption_for_clip(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    for i in range(1, 8):
        text = re.sub(rf"(?i)\bstep\s*{i}\s*:", f"Step {i}:", text)
        text = re.sub(rf"(?i)\bstep{i}\b", f"Step {i}", text)

    text = " ".join(line.strip() for line in text.split("\n") if line.strip())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in [".jpg", ".jpeg", ".png"]


def is_depth_image(path: Path) -> bool:
    return is_image_file(path) and "_depth_suppressed" in path.stem


def is_original_image(path: Path) -> bool:
    return is_image_file(path) and "_depth_suppressed" not in path.stem


class MultiModalRawDataset(Dataset):
    """
    Dataset cho Ver2:
    - Input: ảnh raw + caption raw
    - Output: image tensor + text string + label

    Kỳ vọng cấu trúc:
    - image_root/
        class_a/
            img1.jpg
            img2.jpg
        class_b/
            ...
    - caption_root/
        class_a.json
        class_b.json
        ...

    Mỗi file json có dạng:
    {
        "img1.jpg": {"text": "...", "label": 0},
        "img2.jpg": {"text": "...", "label": 0}
    }
    """

    def __init__(
        self,
        image_root,
        caption_root,
        transform=None,
        use_depth_suppressed: bool = False,
        strict_caption_match: bool = True
    ):
        self.image_root = Path(image_root)
        self.caption_root = Path(caption_root)
        self.transform = transform
        self.use_depth_suppressed = use_depth_suppressed
        self.strict_caption_match = strict_caption_match

        if not self.image_root.exists():
            raise FileNotFoundError(f"Image root not found: {self.image_root}")

        if not self.caption_root.exists():
            raise FileNotFoundError(f"Caption root not found: {self.caption_root}")

        self.caption_db = self._load_caption_db()
        self.samples = self._build_samples()

        if len(self.samples) == 0:
            raise ValueError("No valid multimodal samples found.")

    def _load_caption_db(self) -> Dict[str, Dict[str, dict]]:
        """
        Trả về:
        {
            "Apple_leaf": {
                "img1.jpg": {"text": "...", "label": ...},
                ...
            },
            ...
        }
        """
        caption_db = {}

        json_files = sorted(self.caption_root.glob("*.json"))
        if len(json_files) == 0:
            raise FileNotFoundError(f"No JSON files found in: {self.caption_root}")

        for json_file in json_files:
            class_name = json_file.stem

            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                continue

            caption_db[class_name] = data

        return caption_db

    def _select_image_paths(self) -> List[Path]:
        all_paths = sorted([p for p in self.image_root.rglob("*") if is_image_file(p)])

        if self.use_depth_suppressed:
            return [p for p in all_paths if is_depth_image(p)]
        return [p for p in all_paths if is_original_image(p)]

    def _build_class_to_idx(self, image_paths: List[Path]) -> Dict[str, int]:
        class_names = sorted(list({p.parent.name for p in image_paths}))
        return {cls_name: i for i, cls_name in enumerate(class_names)}

    def _build_samples(self) -> List[dict]:
        image_paths = self._select_image_paths()
        self.class_to_idx = self._build_class_to_idx(image_paths)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        samples = []
        skipped_missing_caption = 0
        skipped_invalid_caption = 0

        for image_path in image_paths:
            class_name = image_path.parent.name
            image_name = image_path.name

            class_caption_map = self.caption_db.get(class_name, None)

            if class_caption_map is None:
                if self.strict_caption_match:
                    skipped_missing_caption += 1
                    continue
                else:
                    text = class_name.replace("_", " ")
                    label = self.class_to_idx[class_name]
                    samples.append({
                        "image_path": image_path,
                        "image_name": image_name,
                        "class_name": class_name,
                        "label": label,
                        "text": text,
                        "source_json": None
                    })
                    continue

            record = class_caption_map.get(image_name, None)

            if record is None or not isinstance(record, dict):
                if self.strict_caption_match:
                    skipped_missing_caption += 1
                    continue
                else:
                    text = class_name.replace("_", " ")
                    label = self.class_to_idx[class_name]
                    samples.append({
                        "image_path": image_path,
                        "image_name": image_name,
                        "class_name": class_name,
                        "label": label,
                        "text": text,
                        "source_json": f"{class_name}.json"
                    })
                    continue

            caption = record.get("text", None)
            label = record.get("label", self.class_to_idx[class_name])

            if caption is None or not isinstance(caption, str) or not caption.strip():
                skipped_invalid_caption += 1
                if self.strict_caption_match:
                    continue
                caption = class_name.replace("_", " ")

            caption = normalize_caption_for_clip(caption)

            samples.append({
                "image_path": image_path,
                "image_name": image_name,
                "class_name": class_name,
                "label": int(label) if label is not None else self.class_to_idx[class_name],
                "text": caption,
                "source_json": f"{class_name}.json"
            })

        print(f"[MultiModalRawDataset] Total selected images: {len(image_paths)}")
        print(f"[MultiModalRawDataset] Valid samples: {len(samples)}")
        print(f"[MultiModalRawDataset] Skipped missing caption: {skipped_missing_caption}")
        print(f"[MultiModalRawDataset] Skipped invalid caption: {skipped_invalid_caption}")
        print(f"[MultiModalRawDataset] Num classes: {len(self.class_to_idx)}")
        print(f"[MultiModalRawDataset] class_to_idx: {self.class_to_idx}")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        image = Image.open(item["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "text": item["text"],
            "label": item["label"],
            "image_name": item["image_name"],
            "image_path": str(item["image_path"].as_posix()),
            "class_name": item["class_name"],
            "source_json": item["source_json"]
        }


def multimodal_raw_collate_fn(batch):
    import torch

    images = torch.stack([item["image"] for item in batch], dim=0)
    texts = [item["text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    return {
        "image": images,
        "text": texts,
        "label": labels,
        "image_name": [item["image_name"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "class_name": [item["class_name"] for item in batch],
        "source_json": [item["source_json"] for item in batch],
    }