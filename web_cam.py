import cv2
import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from collections import deque
import numpy as np
import os

# Definition of TemporalMeanNet (same as in behavior_detection.ipynb)
class TemporalMeanNet(nn.Module):
    def __init__(self, backbone_name: str, n_classes: int):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
            drop_path_rate=0.2,
        )
        self.embed_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(p=0.2),
            nn.Linear(self.embed_dim, n_classes),
        )

    def forward(self, clips):
        b, t, c, h, w = clips.shape
        clips = clips.view(b * t, c, h, w)
        feats = self.backbone(clips)
        feats = feats.view(b, t, -1).mean(dim=1)
        return self.head(feats)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint
script_dir = os.path.dirname(__file__)  # Get the directory of the current script
MODEL_PATH = os.path.join(script_dir, "best_model_convnext_small_in22ft1k_sequence_based_group_split_augmented_no_standing.pth")
ckpt = torch.load(MODEL_PATH, map_location=device)

# Rebuild the model
model = TemporalMeanNet(ckpt["model_name"], len(ckpt["class_names"])).to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# Class names and clip length
class_names = ckpt["class_names"]
CLIP_LEN = ckpt["clip_len"]

print(f"✅ Model loaded: {ckpt['model_name']}")
print(f"Classes: {class_names}")
print(f"Clip length: {CLIP_LEN}")

# Device information
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Memory Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

# Preprocessing (same as valid_tfms in behavior_detection.ipynb)
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# Frame buffer (fixed-length queue)
frame_buffer = deque(maxlen=CLIP_LEN)

# Start webcam
cap = cv2.VideoCapture(2)
# cap = cv2.VideoCapture("http://172.20.10.2/video")

print("Press 'q' to quit.")

torch.backends.cudnn.benchmark = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    
    # Preprocess and add to buffer
    tensor_frame = transform(pil_img)
    frame_buffer.append(tensor_frame)

    # Perform inference when buffer is full
    if len(frame_buffer) == CLIP_LEN:
        # (T,C,H,W) → (1,T,C,H,W)
        clip = torch.stack(list(frame_buffer), dim=0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(clip)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
        pred_idx = probs.argmax()
        pred_label = class_names[pred_idx]
        pred_conf = probs[pred_idx]

        # Draw on screen
        text = f"{pred_label}: {pred_conf:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Behavior Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
