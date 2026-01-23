# -*- coding: utf-8 -*-
"""
Braille dot detection using YOLO model with tiling and NMS.
"""
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from torchvision.ops import nms

# Load model once at module level
MODEL_PATH = "models/best.pt"
model = YOLO(MODEL_PATH)

# Detection parameters
TILE_SIZE = 256
STRIDE = int(TILE_SIZE * 0.75)  # 192 ~25% overlap
CONF_THRESH = 0.5
MAX_ASPECT_RATIO = 1.2  # square tolerance
NMS_IOU = 0.2  # aggressive overlap removal


def detect_dots(image):
    """
    Detect Braille dots from an image using YOLO with tiling approach.
    
    Args:
        image: numpy array (BGR format) of shape (H, W, 3)
        
    Returns:
        list: List of (x, y) tuples representing dot centers
    """
    # Rotate image 180° clockwise
    image = cv2.rotate(image, cv2.ROTATE_180)
    H, W, _ = image.shape
    
    final_boxes = []
    final_scores = []
    final_classes = []
    
    # Tile-based detection
    for y in range(0, H, STRIDE):
        for x in range(0, W, STRIDE):
            tile = image[y:y + TILE_SIZE, x:x + TILE_SIZE]
            
            if tile.shape[0] < 50 or tile.shape[1] < 50:
                continue
            
            tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            
            results = model.predict(tile_rgb, conf=CONF_THRESH, verbose=False)
            
            if results[0].boxes is None:
                continue
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                w = x2 - x1
                h = y2 - y1
                
                # Reject non-square detections
                aspect_ratio = max(w, h) / min(w, h)
                if aspect_ratio > MAX_ASPECT_RATIO:
                    continue
                
                # Convert to square box
                side = max(w, h)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                x1_sq = cx - side / 2
                y1_sq = cy - side / 2
                x2_sq = cx + side / 2
                y2_sq = cy + side / 2
                
                # Map back to original image coordinates
                final_boxes.append([
                    x1_sq + x,
                    y1_sq + y,
                    x2_sq + x,
                    y2_sq + y
                ])
                final_scores.append(conf)
                final_classes.append(cls)
    
    # Apply Global NMS
    if len(final_boxes) == 0:
        return []
    
    boxes = torch.tensor(final_boxes, dtype=torch.float32)
    scores = torch.tensor(final_scores)
    
    keep = nms(boxes, scores, iou_threshold=NMS_IOU)
    
    final_boxes = boxes[keep].numpy()
    final_scores = scores[keep].numpy()
    final_classes = np.array(final_classes)[keep.numpy()]
    
    # Rotate boxes back 180°
    rotated_back_boxes = []
    for box in final_boxes:
        x1, y1, x2, y2 = box
        x1r = W - x2
        y1r = H - y2
        x2r = W - x1
        y2r = H - y1
        rotated_back_boxes.append([x1r, y1r, x2r, y2r])
    
    final_boxes = np.array(rotated_back_boxes)
    
    # Convert boxes to dot centers
    dots = []
    for box in final_boxes:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        dots.append((cx, cy))
    
    return dots
