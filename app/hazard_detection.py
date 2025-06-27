import torch
import torchvision.transforms as T
from PIL import Image
import io

# Load pre-trained object detection model (COCO classes)
model = torch.hub.load('pytorch/vision', 'fasterrcnn_resnet50_fpn', pretrained=True)
model.eval()

# COCO class labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    # Add more if needed
]

# Hazard-related categories to detect
HAZARD_CLASSES = {'person', 'car', 'bus', 'truck', 'traffic light', 'fire hydrant', 'stop sign'}

# Image pre-processing
transform = T.Compose([
    T.ToTensor(),
])

def detect_hazards(image_bytes, confidence_threshold=0.5):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(img_tensor)[0]

    hazards = []
    for idx in range(len(predictions['boxes'])):
        score = predictions['scores'][idx].item()
        label_idx = predictions['labels'][idx].item()
        label = COCO_INSTANCE_CATEGORY_NAMES[label_idx]

        if score >= confidence_threshold and label in HAZARD_CLASSES:
            box = predictions['boxes'][idx].tolist()
            hazards.append({
                "label": label,
                "score": round(score, 2),
                "box": box  # [x1, y1, x2, y2]
            })

    return {
        "hazards_detected": hazards,
        "count": len(hazards)
    }