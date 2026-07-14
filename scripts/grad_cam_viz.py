import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Try importing pytorch_grad_cam, if not available use custom implementation
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    HAS_GRAD_CAM = True
except ImportError:
    HAS_GRAD_CAM = False

# Setup project root
current_dir = Path(__file__).resolve().parent if '__file__' in locals() else Path.cwd()
PROJECT_ROOT = current_dir
while not (PROJECT_ROOT / 'src').exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.mulco import MulCoEndToEnd
from torchvision import transforms
from transformers import AutoTokenizer

# Mapping
class_to_idx = {
    'Apple_Scab_Leaf': 0, 'Apple_leaf': 1, 'Apple_rust_leaf': 2, 'Bell_pepper_leaf': 3, 
    'Bell_pepper_leaf_spot': 4, 'Blueberry_leaf': 5, 'Cherry_leaf': 6, 'Corn_Gray_leaf_spot': 7, 
    'Corn_leaf_blight': 8, 'Corn_rust_leaf': 9, 'Peach_leaf': 10, 'Potato_leaf_early_blight': 11, 
    'Potato_leaf_late_blight': 12, 'Raspberry_leaf': 13, 'Soyabean_leaf': 14, 
    'Squash_Powdery_mildew_leaf': 15, 'Strawberry_leaf': 16, 'Tomato_Early_blight_leaf': 17, 
    'Tomato_Septoria_leaf_spot': 18, 'Tomato_leaf': 19, 'Tomato_leaf_bacterial_spot': 20, 
    'Tomato_leaf_late_blight': 21, 'Tomato_leaf_mosaic_virus': 22, 'Tomato_leaf_yellow_virus': 23, 
    'Tomato_mold_leaf': 24, 'Tomato_two_spotted_spider_mites_leaf': 25, 'grape_leaf': 26, 
    'grape_leaf_black_rot': 27
}
idx_to_class = {v: k for k, v in class_to_idx.items()}

class MulCoWrapper(nn.Module):
    def __init__(self, model, input_ids, attention_mask):
        super().__init__()
        self.model = model
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def forward(self, images):
        return self.model(images, self.input_ids, self.attention_mask)

# Custom GradCAM if pytorch-grad-cam is not installed
class CustomGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx=None):
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        target = output[0, class_idx]
        target.backward()
        
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-7)
        HAS_GRAD_CAM = False # Force CustomGradCAM
        return cam, class_idx

def generate_gradcam(image_path, caption_text, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Model
    print("Loading model...")
    model = MulCoEndToEnd(num_classes=28).to(device)
    ckpt_path = PROJECT_ROOT / "archive/mulco_depth_aug_cb_focal_gem/best_fine_tuned_model.pth"
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True), strict=False)
    model.eval()
    print("Model loaded successfully.")
    
    # 2. Prepare Inputs
    print("Preparing inputs...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    rgb_img = cv2.imread(str(image_path), 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img_float = np.float32(rgb_img) / 255
    
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    tokens = tokenizer(caption_text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)
    
    # Wrap model
    wrapped_model = MulCoWrapper(model, input_ids, attention_mask)
    
    # Target Layer: You can choose the last layer of image_backbone or fusion_blocks
    target_layer = wrapped_model.model.fusion_blocks[-1].restormer.norm2 # Try this layer
    
    print("Running Grad-CAM...")    
    if HAS_GRAD_CAM:
        cam = GradCAM(model=wrapped_model, target_layers=[target_layer])
        targets = None # Predict highest scoring class
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Get prediction
        with torch.no_grad():
            preds = wrapped_model(input_tensor)
            pred_class = preds.argmax(dim=1).item()
            
        visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
    else:
        cam_generator = CustomGradCAM(wrapped_model, target_layer)
        grayscale_cam, pred_class = cam_generator(input_tensor)
        
        heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255
        visualization = heatmap + np.float32(rgb_img_float)
        visualization = visualization / np.max(visualization)
        visualization = np.uint8(255 * visualization)
        
    class_name = idx_to_class.get(pred_class, "Unknown")
    
    # Save Image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title(f"Grad-CAM (Pred: {class_name})")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Grad-CAM saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--caption", required=True, help="Text caption for the image")
    parser.add_argument("--output", default="gradcam_output.png", help="Path to save output")
    args = parser.parse_args()
    
    generate_gradcam(args.image, args.caption, args.output)
