import os
import random
import shutil

# ==========================================
# CẤU HÌNH THÔNG SỐ (HYPERPARAMETERS)
# ==========================================
val_ratio = 0.2  # Tỷ lệ chia: 20% cho tập validation
min_val_images = 2  # Số lượng ảnh TỐI THIỂU cho mỗi lớp ở tập val

# Đặt random seed để kết quả chia không bị thay đổi nếu chạy lại nhiều lần
random.seed(42)

# ==========================================
# KHAI BÁO ĐƯỜNG DẪN ĐỘNG (DYNAMIC PATH)
# ==========================================
# Tự động lấy đường dẫn từ file này ngược ra thư mục gốc
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))

base_dir = os.path.join(project_root, "data", "AIDG", "dataset_PlantDoc", "images")
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# Tạo thư mục val nếu chưa có
os.makedirs(val_dir, exist_ok=True)

# ==========================================
# TIẾN HÀNH CHIA DỮ LIỆU
# ==========================================
print(f"Bắt đầu chia dữ liệu (Tỷ lệ: {val_ratio*100}%, Tối thiểu: {min_val_images} ảnh/lớp)...")
print("-" * 50)

# Duyệt qua từng class (thư mục con) trong tập train
for class_name in os.listdir(train_dir):
    class_train_path = os.path.join(train_dir, class_name)
    
    # Bỏ qua nếu không phải là thư mục
    if not os.path.isdir(class_train_path):
        continue
        
    class_val_path = os.path.join(val_dir, class_name)
    os.makedirs(class_val_path, exist_ok=True)
    
    # Lấy danh sách file hợp lệ
    images = [f for f in os.listdir(class_train_path) if os.path.isfile(os.path.join(class_train_path, f))]
    total_images = len(images)
    
    # Bỏ qua nếu thư mục rỗng
    if total_images == 0:
        continue
        
    random.shuffle(images)
    
    # 1. Tính toán số lượng ảnh đưa vào val theo tỷ lệ
    num_val_images = int(total_images * val_ratio)
    
    # 2. Áp dụng Minimum Threshold
    if num_val_images < min_val_images:
        num_val_images = min_val_images
        
    # 3. Chốt chặn an toàn: Phải giữ lại ít nhất 1 ảnh cho tập train
    # Nếu số ảnh mang đi val lớn hơn hoặc bằng tổng số ảnh đang có
    if num_val_images >= total_images:
        num_val_images = total_images - 1 
        
    # Tiến hành cắt (move) ảnh sang val
    val_images = images[:num_val_images]
    for img in val_images:
        src_path = os.path.join(class_train_path, img)
        dst_path = os.path.join(class_val_path, img)
        shutil.move(src_path, dst_path)
        
    print(f"Lớp '{class_name}': Chuyển {num_val_images} val | Giữ lại {total_images - num_val_images} train (Tổng: {total_images})")

print("-" * 50)
print("✅ Hoàn tất việc tạo tập validation!")