from contextlib import asynccontextmanager
from typing import AsyncGenerator

import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np

from scripts.preprocessing import preprocess_uploads
from scripts.segmentation import create_overlay, compute_segmentation_stats

MODEL_PATH = 'models/oncer_model.keras'
model: tf.keras.Model | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None, None]:
    """
    Load model once at startup.
    """
    global model
    try:    
       print('Loading model...')
       model = tf.keras.models.load_model(MODEL_PATH, compile=False) 
       print('Model loaded successfully!')
       yield
    finally:
       print('Shutting down microservice...')
        

app = FastAPI()


@app.post('/predict')
async def predictTumors(
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
    flair: UploadFile = File(...),
) -> dict[str, object]:
    """
    Attempt brain tumour segmentation on user inputs.
    """
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded...')

    try:
        # Convert to shape (1, H, W, 4)
        input_tensor = await preprocess_uploads(t1, t1ce, t2, flair)

        # Run inference and send response 
        seg_logits = model.predict(input_tensor)
        seg_mask = np.argmax(seg_mask.squeeze(), axis=-1) # shape (H, W)
        seg_mask = np.argmax(seg_logits[0], axis=-1) # shape (1, H, W, 4) -> (H, W, 4) -> (H, W), most likely class per pixel
        tumor_present = bool(np.any(seg_mask > 0))

        overlay = create_overlay(seg_mask)
        stats = compute_segmentation_stats(seg_mask)

        return {
            'tumor_present': tumor_present,
            'overlay': overlay,
            'stats': stats, 
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error during inference: {str(e)}')
    