import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import io
import base64
import torch.nn as nn
import cv2
import numpy as np
import mediapipe as mp

# --- CONFIGURATION ---
IMG_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. DEFINE MODEL ARCHITECTURE ---
# (Must match your training exactly)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x): return self.net(x)

# Load Model
model = CNN().to(device)
try:
    state_dict = torch.load("eye_classifier_finetuned.pth", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# --- 2. SETUP MEDIAPIPE ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Indices for MediaPipe Face Mesh
# Left Eye
LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380]
# Right Eye
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380, 33, 160, 158, 133, 153, 144] 

# --- 3. PREPROCESSING ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

def base64_to_cv2(base64_str):
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    image_bytes = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def crop_eye(image, landmarks, indices):
    """Crops the eye with some padding to match MRL dataset style"""
    h, w, _ = image.shape
    
    # Extract just the eye coordinates
    eye_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    
    # Get bounding box of the eye
    xs = [p[0] for p in eye_points]
    ys = [p[1] for p in eye_points]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    
    # Add Padding (MRL dataset usually has a bit of context around the eye)
    w_eye = x2 - x1
    h_eye = y2 - y1
    pad_x = int(w_eye * 0.3) # 30% padding
    pad_y = int(h_eye * 0.6) # 60% padding (need eyebrow/lid context)

    x1 = max(0, x1 - pad_x)
    x2 = min(w, x2 + pad_x)
    y1 = max(0, y1 - pad_y)
    y2 = min(h, y2 + pad_y)
    
    return image[y1:y2, x1:x2]

# --- 4. API SERVER ---
app = Flask(__name__)
CORS(app)

@app.route("/prediction", methods=["POST"])
def prediction():
    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        # Decode Image
        img_cv = base64_to_cv2(data["image"])
        h, w, _ = img_cv.shape
        
        # Run MediaPipe
        results = face_mesh.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return jsonify({"prediction": None, "message": "No face detected"})

        # Get Landmarks
        landmarks = results.multi_face_landmarks[0].landmark

        # --- A. Calculate Face BBox for UI (Green Square) ---
        x_min = min([lm.x for lm in landmarks]) * w
        y_min = min([lm.y for lm in landmarks]) * h
        x_max = max([lm.x for lm in landmarks]) * w
        y_max = max([lm.y for lm in landmarks]) * h
        bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]

        # --- B. Crop & Predict Eyes ---
        # 1. Crop
        left_crop = crop_eye(img_cv, landmarks, LEFT_EYE_IDXS)
        right_crop = crop_eye(img_cv, landmarks, RIGHT_EYE_IDXS)
        
        preds = []
        confs = []

        # 2. Loop through crops
        for crop in [left_crop, right_crop]:
            if crop.size == 0: continue
            
            # Convert to PIL for Transforms
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            
            # Preprocess (Resize -> Grayscale -> Tensor)
            tensor = transform(pil_img).unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                output = model(tensor)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1).item() # 0=Closed, 1=Open
                conf = probs[0][pred].item()
                
                preds.append(pred)
                confs.append(conf)

        # --- C. Final Logic ---
        # If any eye was successfully detected
        if preds:
            # 0 = Closed, 1 = Open
            # If ANY eye is Open (1), status is Open. Both must be Closed (0) to trigger.
            # (You can flip this logic if you want strict safety)
            if 1 in preds:
                final_label = "Open"
            else:
                final_label = "Closed"
                
            avg_conf = sum(confs) / len(confs)
        else:
            final_label = "Open"
            avg_conf = 0.0

        return jsonify({
            "prediction": final_label,
            "confidence": round(avg_conf, 3),
            "bbox": bbox 
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)