import os
# Hardcoded to use modern Keras 3, not Keras 2 (tf-keras)
os.environ['TF_USE_LEGACY_KERAS'] = '0'
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import traceback

from dotenv import load_dotenv
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np

# from .scripts.model import DiceMetric, dice_loss 
from .scripts.preprocessing import preprocess_uploads
from .scripts.segmentation import create_overlay, compute_segmentation_stats

load_dotenv()
APP_PORT = os.environ.get('APP_PORT', '5173')

MODEL_PATH = Path(__file__).resolve().parent.parent / 'models' / 'oncer_model.keras' 
# MODEL_PATH = Path(os.environ.get(
#     'MODEL_PATH',
#     str(Path(__file__).resolve().parent.parent / 'models' / 'oncer_model_keras3.keras')
# ))
model: tf.keras.Model | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Load model once at startup.
    """
    global model
    try:    
       print('Loading model...')
       model = tf.keras.models.load_model(
            MODEL_PATH, 
            # custom_objects={
            #     'DiceMetric': DiceMetric,
            #     'dice_loss': dice_loss
            # }, 
            compile=False
        ) 
       print('Model loaded successfully!')
       yield
    finally:
       print('Shutting down microservice...')
        

app = FastAPI(lifespan=lifespan)
app_port = os.environ.get('APP_PORT', '5173')
app.add_middleware(
    CORSMiddleware,
    allow_origins=[f'http://localhost:{APP_PORT}'], 
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
app.mount("/papers", StaticFiles(directory="docs"), name="papers")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


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
        seg_mask = np.argmax(seg_logits[0], axis=-1) # shape (1, H, W, 4) -> (H, W, 4) -> (H, W), most likely class per pixel
        # tumor_present = bool(np.any(seg_mask > 0))
        prediction = 'Tumour detected' if bool(np.any(seg_mask > 0)) else 'No tumours detected'

        overlay = create_overlay(seg_mask)
        stats = compute_segmentation_stats(seg_mask)

        return {
            # 'tumor_present': tumor_present,
            'prediction': prediction,
            'overlay': overlay,
            'stats': stats, 
        }
    except Exception:
        traceback.print_exc()
        raise 
    