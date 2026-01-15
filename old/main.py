"""
Braille Detection API (FastAPI) — Full pipeline integration

Steps:
1. Put your YOLO model file `best.pt` in this folder.
2. Install dependencies:
   pip install fastapi uvicorn[standard] pillow numpy opencv-python ultralytics scikit-image python-multipart aiofiles
3. Run:
   uvicorn braille_detection_api:app --host 0.0.0.0 --port 8000

This version integrates:
- Preprocessing (adaptive/Sauvola binarization, morphological cleanup)
- Dot consolidation and detection (connected components)
- YOLO model inference on the cleaned dot image
- Annotated detection output (bounding boxes + dots)

Flutter usage:
Send multipart/form-data with key `file` → receives JPEG annotated image.
Use `Image.memory(response.bodyBytes)` in Flutter.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import io
import base64
import os

# ==========================
# Utility: Dots to Bit + Map
# ==========================
def dots_to_bit(dots):
    val = 0
    for d in dots:
        val |= 1 << (6 - d)
    return val

braille_map = {
    dots_to_bit([1]): 'अ',
    dots_to_bit([3, 4, 5]): 'आ',
    dots_to_bit([2, 4]): 'इ',
    dots_to_bit([3, 5]): 'ई',
    dots_to_bit([1, 3, 6]): 'उ',
    dots_to_bit([1, 2, 5, 6]): 'ऊ',
    dots_to_bit([1, 5, 6]): 'ऋ',
    dots_to_bit([1, 5]): 'ए',
    dots_to_bit([3, 4]): 'ऐ',
    dots_to_bit([1, 3, 5]): 'ओ',
    dots_to_bit([2, 4, 6]): 'औ',
    dots_to_bit([1, 6]): 'अं',
    dots_to_bit([6]): 'अः',
}

# ==========================
# Preprocessing + Dot Detect
# ==========================
def preprocess_braille_image(img_gray, method="adaptive", block_size=31, C=10):
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    if method == "adaptive":
        bin_img = cv2.adaptiveThreshold(
            img_blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size, C)
    else:
        from skimage.filters import threshold_sauvola
        t = threshold_sauvola(img_blur, window_size=block_size, k=0.2)
        bin_img = (img_blur < t).astype(np.uint8) * 255

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel_open)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
    return cleaned

def consolidate_braille_dots(binary_img):
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    blurred = cv2.GaussianBlur(opened, (3, 3), 0)
    _, smooth_bin = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return smooth_bin

def detect_braille_dots(binary_img, min_area=10, max_area=220):
    centroids = []
    num_labels, labels, stats, ctds = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            x, y = ctds[i]
            centroids.append((float(x), float(y)))
    return np.array(centroids, dtype=np.float32)

# ==========================
# FastAPI setup
# ==========================
app = FastAPI(title="Braille Detection API (Full Pipeline)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.environ.get('BRAILLE_MODEL_PATH', 'best_1.pt')
DEVICE = os.environ.get('DEVICE', 'cpu')
model = YOLO(MODEL_PATH)

def read_imagefile_to_bgr(data: bytes):
    image = Image.open(io.BytesIO(data)).convert('RGB')
    arr = np.array(image)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr

def bgr_to_jpeg_bytes(bgr_img):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format='JPEG', quality=90)
    return buf.getvalue()

@app.post('/detect')
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        bgr = read_imagefile_to_bgr(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Step 1. Preprocess and clean
    pre = preprocess_braille_image(gray, method="sauvola")
    binary = consolidate_braille_dots(pre)

    # Step 2. Detect dots
    dots = detect_braille_dots(binary)

    # Step 3. Generate dot-only synthetic image (white background)
    h, w, _ = bgr.shape
    output_img = np.ones((h, w, 3), dtype=np.uint8) * 255
    for (x, y) in dots:
        cv2.circle(output_img, (int(x), int(y)), 4, (0, 0, 0), -1)

    # Step 4. YOLO inference
    results = model(output_img)
    annotated = results[0].plot()

    # Optional overlay detected dots on annotated image
    for (x, y) in dots:
        cv2.circle(annotated, (int(x), int(y)), 3, (0, 0, 255), -1)

    jpeg_bytes = bgr_to_jpeg_bytes(annotated)
    return Response(content=jpeg_bytes, media_type='image/jpeg')

@app.get('/ping')
async def ping():
    return {"status": "ok", "model_loaded": True}
