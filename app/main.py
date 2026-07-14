# -*- coding: utf-8 -*-
"""
FastAPI application for Braille to text conversion (OBR - Object to Braille Recognition).
Accepts images of size 1500x2000px and returns converted text.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import logging

from detector import detect_dots
from grid import braille_grid_detection
from decoder import decode_braille_lines, render_nepali_text, braille_map_text, braille_map_number, half_consonant_symbol_map
# from gemini_client import clean_braille_text
# from gemini_client import clean_braille_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Nepali Braille OBR API",
    description="Object to Braille Recognition API for converting Braille images to text",
    version="1.0.1"
)

# Expected image dimensions
EXPECTED_WIDTH = 1500
EXPECTED_HEIGHT = 2000


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Nepali Braille OBR API",
        "expected_image_size": f"{EXPECTED_WIDTH}x{EXPECTED_HEIGHT}px"
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Detect Braille dots and convert to text.
    
    Args:
        file: Image file (should be 1500x2000px for best results)
        
    Returns:
        JSON response with detected text and metadata
    """
    try:
        # Read image file
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file. Could not decode image.")
        
        # Resize to expected dimensions (1500x2000)
        image = cv2.resize(image, (EXPECTED_WIDTH, EXPECTED_HEIGHT))
        logger.info(f"Processing image of size: {image.shape}")
        
        # Step 1: Detect dots
        dots = detect_dots(image)
        logger.info(f"Detected {len(dots)} dots")
        
        if len(dots) == 0:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "dot_count": 0,
                    "text": "",
                    "message": "No Braille dots detected in the image."
                }
            )
        
        # Step 2: Grid detection and binary conversion
        theta=5.0
        D_dis=3.0
        _, braille_bins = braille_grid_detection(dots, theta, D_dis, image)
        logger.info(f"Detected {len(braille_bins)} text lines")
        
        # Step 3: Decode to text
        text = decode_braille_lines(braille_bins, braille_map_text, braille_map_number)
        obr_text = render_nepali_text(text, half_consonant_symbol_map)

        # Step 4: Post-process with Gemini (if API key configured)

        #no longer using gemini for post-processing
        # cleaned_text = clean_braille_text(text)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "dot_count": len(dots),
                "obr_text": obr_text,
                "clean_text": obr_text,
                # "clean_text": cleaned_text or text,
                # "used_gemini": cleaned_text is not None,
                "lines_detected": len(braille_bins),
                "image": "/temp/viz.jpg"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint for monitoring/ALB."""
    return {"status": "healthy"}
