import io
import traceback

from fastapi import UploadFile, File, HTTPException
import tensorflow as tf
import numpy as np
from PIL import Image

from src.config import IMG_SIZE, MODALITIES


def load_and_normalize(upload_file: UploadFile) -> np.ndarray:
    """
    Read uploads and normalize pixel intensities. 
    """
    try: 
        contents = upload_file.file.read()
        upload_file.file.seek(0) # reset pointer (back to beginning of file)
        img = Image.open(io.BytesIO(contents)).convert('L') # bytes -> in-memory file-like obj -> greyscale-ify (if not already)
        img = img.resize(IMG_SIZE)

        min_val = np.min(img)
        max_val = np.max(img)

        if max_val > min_val: # has contrast
            img = (img - min_val) / (max_val - min_val) # scale to [0, 1]
        else: # has no contrast
            img = np.zeros_like(img) # handle corrupted/missing values   
        return np.array(img, dtype=np.float32)
    except Exception:
        traceback.print_exc()
        raise 
    
        raise HTTPException(status_code=400, detail=f'Invalid image file: {upload_file.filename}')


async def preprocess_uploads(
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
    flair: UploadFile = File(...),
) -> tf.Tensor:   
    """ 
    Load, normalize, and stack MRI modalities into a model-ready tensor.  
    """
    t1_img = load_and_normalize(t1)
    t1ce_img = load_and_normalize(t1ce)
    t2_img = load_and_normalize(t2)
    flair_img = load_and_normalize(flair)


    modality_imgs = {
        't1': t1_img,
        't1ce': t1ce_img,
        't2': t2_img,
        'flair': flair_img,
    }
    stacked = np.stack([modality_imgs[m] for m in MODALITIES], axis=-1)

    # stacked = np.stack([locals()[m + '_img'] for m in MODALITIES], axis=-1) # shape (H, W) x 4 -> (H, W, 4) (append new dim at the end)
    input_tensor = np.expand_dims(stacked, axis=0) # shape (H, W, 4) -> (1, H, W, 4)
    return tf.convert_to_tensor(input_tensor, dtype=tf.float32)
