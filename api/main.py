import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
import io

try:
    from pytorch_grad_cam import GradCAM
    HAS_GRAD_CAM = True
except ImportError:
    HAS_GRAD_CAM = False

# Setup project root so we can import src
current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = current_dir.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.mulco import MulCoEndToEnd
from torchvision import transforms
from transformers import AutoTokenizer

app = FastAPI(title="MulCo-PlantNet Web Demo API")

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIG ---
import os
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_PROMPT = """
Please describe the image of the plant leaf according to the following guidelines:
Step1,Identify the color of the leaf, including the base color and any phenotypic characteristics of spots or discolored areas, such as their location, size, length, number, and color.
Step2,describe the overall shape of the leaf, including whether it is round, oval, heart-shaped, or another shape.
Step3,describe the texture of the leaf, including whether the surface is smooth, hairy, or has other features.
Step4,if the leaf has edges, describe the characteristics of the edges, such as whether they are smooth, serrated, or wavy.
Step5,describe whether there are any visible damages on the leaf, such as holes, tears, wilting, or lesions.
Step6,if there are veins on the leaf, describe their distribution and color.
Step7,if the image includes the petiole of the leaf, describe the thickness, color, and texture of the petiole.
Please keep the description objective, providing only the information visible in the image without adding any inferences or explanations.
"""

# Mapping
class_to_idx = {
    'Apple_leaf': 0, 'Apple_rust_leaf': 1, 'Apple_Scab_Leaf': 2, 'Bell_pepper_leaf': 3, 
    'Bell_pepper_leaf_spot': 4, 'Blueberry_leaf': 5, 'Cherry_leaf': 6, 'Corn_Gray_leaf_spot': 7, 
    'Corn_leaf_blight': 8, 'Corn_rust_leaf': 9, 'grape_leaf': 10, 'grape_leaf_black_rot': 11, 
    'Peach_leaf': 12, 'Potato_leaf_early_blight': 13, 'Potato_leaf_late_blight': 14, 
    'Raspberry_leaf': 15, 'Soyabean_leaf': 16, 'Squash_Powdery_mildew_leaf': 17, 'Strawberry_leaf': 18, 
    'Tomato_Early_blight_leaf': 19, 'Tomato_leaf': 20, 'Tomato_leaf_bacterial_spot': 21, 
    'Tomato_leaf_late_blight': 22, 'Tomato_leaf_mosaic_virus': 23, 'Tomato_leaf_yellow_virus': 24, 
    'Tomato_mold_leaf': 25, 'Tomato_Septoria_leaf_spot': 26, 'Tomato_two_spotted_spider_mites_leaf': 27
}
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Globals
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class MulCoWrapper(nn.Module):
    def __init__(self, model, input_ids, attention_mask):
        super().__init__()
        self.model = model
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def forward(self, images):
        return self.model(images, self.input_ids, self.attention_mask)

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
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
        return cam, class_idx

@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    print(f"Loading model on {device}...")
    model = MulCoEndToEnd(num_classes=28).to(device)
    ckpt_path = PROJECT_ROOT / "archive/mulco_depth_aug_cb_focal_gem/best_fine_tuned_model.pth"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True), strict=False)
        model.eval()
        print("Model loaded successfully.")
    else:
        print(f"Warning: Checkpoint not found at {ckpt_path}")
    
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    print("Tokenizer loaded.")

def generate_caption_gemini(pil_image: Image.Image) -> str:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[GEMINI_PROMPT, pil_image]
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "A picture of a leaf"

def apply_colormap_on_image(org_im, activation, colormap_name=cv2.COLORMAP_JET):
    # org_im should be float32 in [0, 1], RGB
    # activation should be float32 in [0, 1]
    heatmap = cv2.applyColorMap(np.uint8(255 * activation), colormap_name)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    cam = heatmap + org_im
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

@app.post("/predict")
async def predict(file: UploadFile = File(...), caption: str = Form(None)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    generated_caption = False
    caption_text = caption
    if not caption_text or caption_text.strip() == "":
        print("Caption empty. Generating with Gemini...")
        caption_text = generate_caption_gemini(image)
        generated_caption = True
        
    print(f"Using caption: {caption_text}")
    
    # Preprocess image
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Preprocess text
    tokens = tokenizer(caption_text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)
    
    # Wrap model for GradCAM
    wrapped_model = MulCoWrapper(model, input_ids, attention_mask)
    target_layer = wrapped_model.model.fusion_blocks[-1].restormer.ffn.project_out
    
    # Inference and GradCAM
    if HAS_GRAD_CAM:
        cam_extractor = GradCAM(model=wrapped_model, target_layers=[target_layer])
        grayscale_cam = cam_extractor(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        
        with torch.no_grad():
            outputs = wrapped_model(input_tensor)
            class_idx = outputs.argmax(dim=1).item()
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence = probabilities[0][class_idx].item()
    else:
        cam_extractor = CustomGradCAM(model=wrapped_model, target_layer=target_layer)
        grayscale_cam, class_idx = cam_extractor(input_tensor)
        
        with torch.no_grad():
            outputs = wrapped_model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence = probabilities[0][class_idx].item()
    
    prediction_class = idx_to_class[class_idx]
    
    # Generate Heatmap image
    rgb_img = np.array(image.resize((224, 224)))
    rgb_img_float = np.float32(rgb_img) / 255
    
    cam_image = apply_colormap_on_image(rgb_img_float, grayscale_cam)
    cam_pil = Image.fromarray(cam_image)
    
    # Convert heatmap to base64
    buffered = io.BytesIO()
    cam_pil.save(buffered, format="JPEG")
    cam_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {
        "prediction": prediction_class,
        "confidence": confidence,
        "caption": caption_text,
        "generated_caption": generated_caption,
        "gradcam_base64": f"data:image/jpeg;base64,{cam_base64}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
