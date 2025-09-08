import io
import asyncio

from fastapi import UploadFile
import numpy as np
from PIL import Image, ImageOps

from ..utils.config import IMG_SIZE, EXTENSIONS


def enhance_contrast(pii_img: Image.Image) -> Image.Image:
   '''Improve tumour visibility in low-contrast MRI uploads'''
   return ImageOps.autocontrast(pii_img)


async def read_image(file: UploadFile) -> np.ndarray:
    '''single file -> numpy array'''
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('L') # greyscale-ify (if not already)
    img = ImageOps.autocontrast(img) # improve tumour visibility in low-contrast MRI uploads
    img = img.resize(*IMG_SIZE)
    return np.array(img, dtype=np.float32) / 255.0 # normalize


# async def load_and_resize(file: UploadFile) -> np.ndarray:
#   '''Takes a single image upload and '''
#   contents = await file.read()
#   img = Image.open(io.BytesIO(contents)).convert('L') # greyscale-ify
#   img = img.resize((240, 240))
#   return np.array(img, dtype=np.float32) / 255.0 # normalize


# async def preprocess_modalities(t1, t1ce, t2, flair) -> np.ndarray:
#   imgs = await asyncio.gather(
#     load_and_resize(t1),
#     load_and_resize(t1ce),
#     load_and_resize(t2),
#     load_and_resize(flair)
#   )

#   stacked = np.stack(imgs, axis=-1) # shape: (240, 240, 4)
#   return np.expand_dims(stacked, axis=0) # shape: (1, 240, 240, 4)

async def extract_modalities_from_zip(file: UploadFile) -> 

async def preprocess_upload(file: UploadFile) -> np.ndarray:
    # imgs = await extract_modalities_from_zip(file) if file.filename.endswith('.zip') else await read_image(file)

    try:
        


    return