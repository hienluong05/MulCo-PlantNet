import json
import re
import numpy as np
import torch
import open_clip
from pathlib import Path
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 1. PATH CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

caption_root = PROJECT_ROOT / "data" / "AIDG" / "captions_LLaVA" / "train"
output_root = PROJECT_ROOT / "data" / "AIDG" / "encoded_text"

embedding_path = output_root / "train_text_embeddings_clip_vitl14.npy"
metadata_path = output_root / "train_text_embeddings_clip_vitl14_metadata.json"

print("PROJECT_ROOT:", PROJECT_ROOT)
print("Caption root:", caption_root)
print("Embedding path:", embedding_path)
print("Metadata path:", metadata_path)

output_root.mkdir(parents=True, exist_ok=True)

# =========================
# 2. HELPERS
# =========================
def normalize_caption_for_clip(text: str) -> str:
    """
    Chuẩn hóa caption để đưa vào CLIP:
    - bỏ khoảng trắng thừa
    - đổi xuống dòng thành khoảng trắng
    - chuẩn hóa 'Step1:' -> 'Step 1:'
    """
    text = (text or "").strip()

    # chuẩn hóa newline
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # chuẩn hóa Step1 -> Step 1
    for i in range(1, 8):
        text = re.sub(rf"(?i)\bstep\s*{i}\s*:", f"Step {i}:", text)
        text = re.sub(rf"(?i)\bstep{i}\b", f"Step {i}", text)

    # gộp nhiều dòng thành một dòng
    text = " ".join(line.strip() for line in text.split("\n") if line.strip())

    # gộp khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================
# 3. LOAD ALL CAPTIONS
# =========================
json_files = sorted(caption_root.glob("*.json"))
print(f"Found {len(json_files)} class json files.")

if len(json_files) == 0:
    raise FileNotFoundError(f"No JSON files found in: {caption_root}")

all_items = []

for json_file in json_files:
    class_name = json_file.stem

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        print(f"[WARN] {json_file.name} is not a dict. Skipping.")
        continue

    for image_name, record in data.items():
        if not isinstance(record, dict):
            continue

        caption = record.get("text", None)
        label = record.get("label", None)

        if caption is None or not isinstance(caption, str) or not caption.strip():
            continue

        caption = normalize_caption_for_clip(caption)

        item = {
            "index": len(all_items),
            "image_name": image_name,
            "class_name": class_name,
            "label": label,
            "caption": caption,
            "source_json": json_file.name
        }
        all_items.append(item)

print(f"Collected {len(all_items)} caption records.")

if len(all_items) == 0:
    raise ValueError("No valid captions found.")

captions = [item["caption"] for item in all_items]

# =========================
# 4. LOAD CLIP MODEL
# =========================
print("Loading CLIP ViT-L/14...")
model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-L-14")
model = model.to(device).eval()

# =========================
# 5. ENCODE TEXT
# =========================
batch_size = 64
all_embeddings = []

print("Encoding captions with CLIP...")
with torch.no_grad():
    for i in tqdm(range(0, len(captions), batch_size), desc="Encoding text"):
        batch_texts = captions[i:i + batch_size]
        tokens = tokenizer(batch_texts).to(device)

        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)

        all_embeddings.append(feats.cpu())

all_embeddings = torch.cat(all_embeddings, dim=0).numpy()

# =========================
# 6. SAVE EMBEDDINGS
# =========================
np.save(embedding_path, all_embeddings)

# =========================
# 7. SAVE METADATA
# =========================
metadata = []
for idx, item in enumerate(all_items):
    metadata.append({
        "index": idx,
        "image_name": item["image_name"],
        "class_name": item["class_name"],
        "label": item["label"],
        "caption": item["caption"],
        "source_json": item["source_json"]
    })

with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

# =========================
# 8. DONE
# =========================
print("Saved embeddings to:", embedding_path.resolve())
print("Saved metadata to:", metadata_path.resolve())
print("Embedding exists:", embedding_path.exists())
print("Metadata exists:", metadata_path.exists())
print("Embedding shape:", all_embeddings.shape)
print("Total metadata records:", len(metadata))