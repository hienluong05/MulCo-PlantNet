import argparse
import hashlib
import json
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ImageSample:
    source_root: Path
    class_name: str
    image_path: Path


def normalize_caption(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = " ".join(line.strip() for line in text.split("\n") if line.strip())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_caption_pool(caption_roots: List[Path]) -> Dict[str, List[str]]:
    pool: Dict[str, List[str]] = {}
    for root in caption_roots:
        if not root.exists():
            continue
        for json_file in sorted(root.glob("*.json")):
            class_name = json_file.stem
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            captions = []
            if isinstance(data, dict):
                for item in data.values():
                    if isinstance(item, dict):
                        text = normalize_caption(str(item.get("text", "")))
                        if text:
                            captions.append(text)
            if captions:
                pool.setdefault(class_name, []).extend(captions)
    return pool


def list_images_from_roots(image_roots: List[Path]) -> Dict[str, List[ImageSample]]:
    by_class: Dict[str, List[ImageSample]] = {}
    for root in image_roots:
        if not root.exists():
            continue
        for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            class_name = class_dir.name
            for img in sorted(class_dir.iterdir()):
                if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
                    by_class.setdefault(class_name, []).append(
                        ImageSample(source_root=root, class_name=class_name, image_path=img)
                    )
    return by_class


def source_tag(path: Path) -> str:
    tag = f"{path.parent.name}_{path.name}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tag)


def ensure_link_or_copy(src: Path, dst: Path) -> str:
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def caption_from_pool(class_name: str, image_name: str, caption_pool: Dict[str, List[str]]) -> Tuple[str, str]:
    class_pool = caption_pool.get(class_name, [])
    if class_pool:
        digest = hashlib.md5(f"{class_name}:{image_name}".encode("utf-8")).hexdigest()
        idx = int(digest, 16) % len(class_pool)
        return class_pool[idx], "llava_pool"

    fallback = class_name.replace("_", " ")
    return f"A close-up image of {fallback} showing visible leaf characteristics.", "template_fallback"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a large multimodal validation set (image + caption) in class-json format."
    )
    parser.add_argument(
        "--image-roots",
        nargs="+",
        default=[
            "data/processed/PlantDocSplited_depth_AUG/validation",
            "data/processed/PlantDocSplited_depth_AUG/test",
            "data/AIDG/dataset_PlantDoc/images/test",
            "data/processed/PlantDocSplited_depth_AUG/train",
        ],
        help="Source image roots (each must contain class subdirectories).",
    )
    parser.add_argument(
        "--caption-roots",
        nargs="+",
        default=[
            "data/AIDG/captions_LLaVA/train",
            "data/AIDG/captions_LLaVA/test",
        ],
        help="Roots containing class_name.json caption files.",
    )
    parser.add_argument(
        "--output-root",
        default="data/processed/PlantDoc_multimodal_validation_large",
        help="Output root for generated images and captions.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=140,
        help="Maximum number of samples per class in generated validation set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible class-wise sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    image_roots = [Path(p) for p in args.image_roots]
    caption_roots = [Path(p) for p in args.caption_roots]

    output_root = Path(args.output_root)
    image_out_root = output_root / "images"
    caption_out_root = output_root / "captions"
    history_path = output_root / "history.json"
    manifest_path = output_root / "manifest.jsonl"

    image_out_root.mkdir(parents=True, exist_ok=True)
    caption_out_root.mkdir(parents=True, exist_ok=True)

    caption_pool = load_caption_pool(caption_roots)
    class_to_samples = list_images_from_roots(image_roots)
    class_names = sorted(class_to_samples.keys())
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    records_by_class: Dict[str, Dict[str, dict]] = {cls: {} for cls in class_names}
    manifest_rows: List[dict] = []

    total_linked = 0
    link_mode_stats = {"hardlink": 0, "copy": 0}

    for class_name in class_names:
        candidates = class_to_samples[class_name]
        if len(candidates) > args.max_per_class:
            selected = random.sample(candidates, args.max_per_class)
        else:
            selected = candidates

        class_image_dir = image_out_root / class_name
        class_image_dir.mkdir(parents=True, exist_ok=True)

        for sample in selected:
            stem = sample.image_path.stem
            ext = sample.image_path.suffix.lower()
            tag = source_tag(sample.source_root)
            unique_name = f"{tag}__{stem}{ext}"
            dst_path = class_image_dir / unique_name

            mode = ensure_link_or_copy(sample.image_path, dst_path)
            link_mode_stats[mode] += 1
            total_linked += 1

            caption_text, caption_source = caption_from_pool(class_name, unique_name, caption_pool)

            records_by_class[class_name][unique_name] = {
                "text": caption_text,
                "label": class_to_idx[class_name],
            }

            manifest_rows.append(
                {
                    "image_name": unique_name,
                    "image_path": str(dst_path.as_posix()),
                    "class_name": class_name,
                    "label": class_to_idx[class_name],
                    "text": caption_text,
                    "caption_source": caption_source,
                    "original_image_path": str(sample.image_path.as_posix()),
                    "source_root": str(sample.source_root.as_posix()),
                    "copy_mode": mode,
                }
            )

    for class_name, payload in records_by_class.items():
        out_file = caption_out_root / f"{class_name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    history = {
        "output_root": str(output_root.as_posix()),
        "num_classes": len(class_names),
        "num_samples": len(manifest_rows),
        "class_to_idx": class_to_idx,
        "max_per_class": args.max_per_class,
        "seed": args.seed,
        "image_roots": [str(p.as_posix()) for p in image_roots],
        "caption_roots": [str(p.as_posix()) for p in caption_roots],
        "link_mode_stats": link_mode_stats,
        "notes": "Captions are borrowed from LLaVA class pools with deterministic hashing; template fallback when class caption is missing.",
    }
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Generated samples: {len(manifest_rows)}")
    print(f"[DONE] Classes: {len(class_names)}")
    print(f"[DONE] Image root: {image_out_root.as_posix()}")
    print(f"[DONE] Caption root: {caption_out_root.as_posix()}")
    print(f"[DONE] Manifest: {manifest_path.as_posix()}")
    print(f"[DONE] History: {history_path.as_posix()}")
    print(f"[DONE] Link stats: {link_mode_stats}")


if __name__ == "__main__":
    main()
