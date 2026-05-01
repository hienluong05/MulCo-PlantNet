import csv
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


def build_caption_key_candidates(image_name: str) -> List[str]:
    """
    Tạo danh sách key caption có thể khớp từ tên ảnh hiện tại.
    Ví dụ:
    - train_Apple leaf_1_depth_suppressed.jpg -> train_Apple leaf_1.jpg
    """
    path_obj = Path(image_name)
    suffix = path_obj.suffix
    stem = path_obj.stem
    candidates = []

    # Ưu tiên key gốc khi ảnh depth được hậu tố _depth_suppressed
    if stem.endswith("_depth_suppressed"):
        base_stem = stem[: -len("_depth_suppressed")]
        candidates.append(f"{base_stem}{suffix}")

    # Fallback: chính tên hiện tại
    candidates.append(image_name)

    # Loại trùng nhưng giữ thứ tự
    unique = []
    seen = set()
    for c in candidates:
        if c not in seen:
            unique.append(c)
            seen.add(c)
    return unique


def _normalize_rel_key(path_text: str) -> str:
    return str(path_text).replace("\\", "/").strip().lower()


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
        strict_caption_match: bool = True,
        image_caption_mapping_path: Optional[str] = None,
    ):
        self.image_root = Path(image_root)
        self.caption_root = Path(caption_root)
        self.transform = transform
        self.use_depth_suppressed = use_depth_suppressed
        self.strict_caption_match = strict_caption_match
        self.image_caption_mapping_path = Path(image_caption_mapping_path) if image_caption_mapping_path else None

        if not self.image_root.exists():
            raise FileNotFoundError(f"Image root not found: {self.image_root}")

        if not self.caption_root.exists():
            raise FileNotFoundError(f"Caption root not found: {self.caption_root}")

        self.caption_db = self._load_caption_db()
        self.image_to_caption_key_map = self._load_image_caption_mapping()
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

    def _load_image_caption_mapping(self) -> Dict[str, str]:
        """
        Trả về map:
        {
            "class_a/image_depth_name.jpg": "caption_key_name.jpg",
            ...
        }
        Hỗ trợ:
        - JSON: {"class_a/image.jpg": "caption.jpg"} hoặc {"class_a": {"image.jpg": "caption.jpg"}}
        - CSV: cột bắt buộc image_name, caption_key; tùy chọn class_name
        """
        if self.image_caption_mapping_path is None:
            return {}

        if not self.image_caption_mapping_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {self.image_caption_mapping_path}")

        suffix = self.image_caption_mapping_path.suffix.lower()
        mapping: Dict[str, str] = {}

        if suffix == ".json":
            with open(self.image_caption_mapping_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("JSON mapping must be an object/dict.")

            for key, value in data.items():
                if isinstance(value, str):
                    mapping[_normalize_rel_key(key)] = value
                elif isinstance(value, dict):
                    class_name = str(key).strip()
                    for image_name, caption_key in value.items():
                        if isinstance(caption_key, str):
                            rel = _normalize_rel_key(f"{class_name}/{image_name}")
                            mapping[rel] = caption_key
        elif suffix == ".csv":
            with open(self.image_caption_mapping_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                required = {"image_name", "caption_key"}
                if not required.issubset(set(reader.fieldnames or [])):
                    raise ValueError("CSV mapping must contain columns: image_name, caption_key (optional: class_name)")
                for row in reader:
                    image_name = (row.get("image_name") or "").strip()
                    caption_key = (row.get("caption_key") or "").strip()
                    class_name = (row.get("class_name") or "").strip()
                    if not image_name or not caption_key:
                        continue
                    rel = _normalize_rel_key(f"{class_name}/{image_name}" if class_name else image_name)
                    mapping[rel] = caption_key
        else:
            raise ValueError("Mapping file must be .json or .csv")

        print(f"[MultiModalRawDataset] Loaded image-caption mapping: {len(mapping)} entries")
        return mapping

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
        mapped_by_file = 0

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

            record = None
            matched_caption_key = None
            rel_key_with_class = _normalize_rel_key(f"{class_name}/{image_name}")
            rel_key_image_only = _normalize_rel_key(image_name)

            mapped_caption_key = self.image_to_caption_key_map.get(rel_key_with_class)
            if mapped_caption_key is None:
                mapped_caption_key = self.image_to_caption_key_map.get(rel_key_image_only)
            if mapped_caption_key is not None:
                candidate_record = class_caption_map.get(mapped_caption_key, None)
                if isinstance(candidate_record, dict):
                    record = candidate_record
                    matched_caption_key = mapped_caption_key
                    mapped_by_file += 1

            if record is None:
                for candidate_key in build_caption_key_candidates(image_name):
                    record = class_caption_map.get(candidate_key, None)
                    if isinstance(record, dict):
                        matched_caption_key = candidate_key
                        break

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
                "source_json": f"{class_name}.json::{matched_caption_key}" if matched_caption_key is not None else f"{class_name}.json"
            })

        print(f"[MultiModalRawDataset] Total selected images: {len(image_paths)}")
        print(f"[MultiModalRawDataset] Valid samples: {len(samples)}")
        print(f"[MultiModalRawDataset] Skipped missing caption: {skipped_missing_caption}")
        print(f"[MultiModalRawDataset] Skipped invalid caption: {skipped_invalid_caption}")
        print(f"[MultiModalRawDataset] Matched by external mapping: {mapped_by_file}")
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