import os
import json
from pathlib import Path

# --- CẤU HÌNH ĐƯỜNG DẪN (Bạn có thể điều chỉnh cho đúng thực tế) ---
IMG_VAL_DIR = "data/AIDG/dataset_PlantDoc/images/val"
CAPTION_TRAIN_DIR = "data/AIDG/captions_LLaVA/train"
CAPTION_VAL_OUT_DIR = "data/AIDG/captions_LLaVA/val"

def extract_val_captions():
    # 1. Tạo thư mục output nếu chưa có
    os.makedirs(CAPTION_VAL_OUT_DIR, exist_ok=True)

    # 2. Lấy danh sách tất cả file ảnh trong tập val (quét cả thư mục con nếu có)
    print(f"--- Đang quét danh sách ảnh trong: {IMG_VAL_DIR} ---")
    val_image_names = set()
    for root, dirs, files in os.walk(IMG_VAL_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                val_image_names.add(file)
    
    print(f"Tìm thấy {len(val_image_names)} ảnh trong tập Validation.")

    # 3. Duyệt qua các file JSON trong thư mục train để tìm caption
    print(f"\n--- Đang trích xuất caption từ: {CAPTION_TRAIN_DIR} ---")
    json_files = [f for f in os.listdir(CAPTION_TRAIN_DIR) if f.endswith('.json')]
    
    total_extracted = 0

    for json_file in json_files:
        train_json_path = os.path.join(CAPTION_TRAIN_DIR, json_file)
        val_json_data = {}

        with open(train_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Kiểm tra từng ảnh trong file JSON
            for img_name, info in data.items():
                if img_name in val_image_names:
                    val_json_data[img_name] = info
        
        # 4. Nếu tìm thấy caption cho tập val trong file JSON này, lưu ra file mới
        if val_json_data:
            output_path = os.path.join(CAPTION_VAL_OUT_DIR, json_file)
            with open(output_path, 'w', encoding='utf-8') as f_out:
                json.dump(val_json_data, f_out, indent=4, ensure_ascii=False)
            
            num_found = len(val_json_data)
            total_extracted += num_found
            print(f"Đã trích xuất {num_found} caption vào: {output_path}")

    print(f"\n Hoàn tất! Tổng cộng đã tìm thấy {total_extracted}/{len(val_image_names)} caption.")
    if total_extracted < len(val_image_names):
        print(" Lưu ý: Một số ảnh validation chưa có caption trong các file JSON tập Train.")

if __name__ == "__main__":
    extract_val_captions()