#imports
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import cv2

# config
MODEL_PATH = 'best.pt'
IMAGE_PATH = Path('License Plate Recognition/valid/images')
OUTPUT_PATH = Path('ocr/valid/images')
CONF = 0.05
PADDING = 0.08

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL_PATH)

# padding helper
def pad_box(x1, y1, x2, y2, pad, w, h):
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * pad)
    py = int(bh * pad)

    x1 = max(0, x1-px)
    y1 = max(0, y1-py)
    x2 = min(w-1, x2 + px)
    y2 = min(h-1, y2 + py)

    return x1, y1, x2, y2

for img_path in IMAGE_PATH.iterdir():
    if img_path.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    h, w = img.shape[:2]

    results = model.predict(source=img, conf=CONF, verbose=False)[0]

    if results.boxes is None:
        continue

    for idx, box in enumerate(results.boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = pad_box(x1, y1, x2, y2, PADDING, w, h)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        out_path = f"{img_path.stem}_plate_{idx}.jpg"
        out_path = OUTPUT_PATH / out_path
        cv2.imwrite(str(out_path), crop)

print("Done!")
