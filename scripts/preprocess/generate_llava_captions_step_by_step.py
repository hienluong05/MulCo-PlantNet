import json
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

# =========================
# 1. CONFIG PATHS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_ROOT = PROJECT_ROOT / "data" / "AIDG" / "dataset_PlantDoc" / "images" / "train"
CAPTION_OUT_ROOT = PROJECT_ROOT / "data" / "AIDG" / "captions_LLaVA" / "train"

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 2. PROMPT TEMPLATE
# =========================
PROMPT_TEMPLATE = """You are an expert botanist and plant pathologist. Carefully examine the image of the plant leaf.
Your task is to provide a highly detailed, step-by-step description of its visual characteristics. 
It is crucial to focus on fine-grained biological features that distinguish closely related species (such as Tomato vs. Potato leaves) and specific disease symptoms.

Provide your description strictly in the following format:
Step 1: Botanical Characteristics. Detail the leaf's overall shape, margin structure (e.g., deeply lobed and serrated vs. entire/smooth), texture, venation pattern, and any visible fine hairs (trichomes).
Step 2: Disease Symptoms. Describe the appearance of spots, lesions, or discolorations. Be highly specific about lesion color, shape, concentric rings (if any), yellow halos (chlorosis), and distribution on the leaf.
Step 3: Background. Briefly describe if the background is uniform, depth-suppressed, or a natural field setting.
Step 4: Final Assessment. Based on the visual evidence, summarize the leaf as belonging to the class '{class_name}'.
"""

def main():
    CAPTION_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading LLaVA model {MODEL_ID} with 4-bit quantization...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    class_dirs = sorted([d for d in IMAGE_ROOT.iterdir() if d.is_dir()])
    
    for idx, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        out_json_path = CAPTION_OUT_ROOT / f"{class_name}.json"
        
        # Load kết quả cũ nếu script từng bị dừng giữa chừng
        results = {}
        if out_json_path.exists():
            with open(out_json_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                
        image_paths = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        
        print(f"Processing class [{idx+1}/{len(class_dirs)}]: {class_name} | Found {len(image_paths)} images")
        
        for img_path in tqdm(image_paths, desc=class_name):
            img_name = img_path.name
            if img_name in results:
                continue  # Bỏ qua nếu đã gen rồi
                
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")
                continue
            
            # Format prompt cho phù hợp với LLaVA
            clean_class_name = class_name.replace("_", " ")
            text_prompt = PROMPT_TEMPLATE.format(class_name=clean_class_name)
            chat_prompt = f"USER: <image>\n{text_prompt}\nASSISTANT:"
            
            inputs = processor(text=chat_prompt, images=image, return_tensors="pt").to(DEVICE, torch.float16)
            
            with torch.no_grad():
                # Sử dụng temperature thấp (0.2) để tránh LLM "ảo giác" (hallucination)
                generate_ids = model.generate(
                    **inputs, 
                    max_new_tokens=256, 
                    temperature=0.2, 
                    do_sample=True
                )
            
            generated_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            # Trích xuất phần text được sinh ra bởi ASSISTANT
            if "ASSISTANT:" in generated_text:
                response = generated_text.split("ASSISTANT:")[-1].strip()
            else:
                response = generated_text.strip()
                
            results[img_name] = {
                "text": response,
                "label": idx
            }
            
        # Lưu ra file json sau khi xong 1 class
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
if __name__ == "__main__":
    main()