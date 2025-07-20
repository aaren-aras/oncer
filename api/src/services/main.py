from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
import numpy as np

from scripts.preprocessing import preprocess_upload
from scripts.segmentation import create_overlay, compute_segmentation_stats

app = FastAPI()
model = tf.keras.models.load_model('models/brain_tumour_multitask.keras') # load on startup

@asynccontextmanager
async def lifespan(app: FastAPI):
    

@app.post('/predict')
async def predictTumours(
  t1: UploadFile = File(...),
  t1ce: UploadFile = File(...),
  t2: UploadFile = File(...),
    flair: UploadFile = File(...),
):
  try:
    # Convert to shape (1, 240, 240, 4)
    input_tensor = await preprocess_modalities(t1, t1ce, t2, flair)

    # Perform predictions
    output = model.predict(input_tensor)
    seg_mask, class_probs = output

    class_probs = class_probs[0]
    seg_mask = np.argmax(seg_mask.squeeze(), axis=-1) # shape: (240, 240)

    prediction = 'Tumour(s) detected' if seg_mask else 'No tumours detected'
    overlay = create_overlay(seg_mask)
    stats = compute_segmentation_stats(seg_mask)

    return {
      'prediction': prediction,
      'overlay': overlay,
      'stats': stats
    }

  
  except Exception as e:
    raise HTTPException(status_code=500, detail=f'Error during inference: {str(e)}')