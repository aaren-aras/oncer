# convert_to_npy.py
import numpy as np
import sys
from PIL import Image

def convert(image_path, npy_path):
    img = Image.open(image_path).convert('L')  # grayscale
    img = img.resize((240, 240))
    img_array = np.array(img).astype(np.float32) / 255.0

    # Put into 4-channel format (T1 = image, others = zeros)
    out = np.zeros((240, 240, 4), dtype=np.float32)
    out[:, :, 0] = img_array  # channel 0 = T1
    np.save(npy_path, out)

if __name__ == '__main__':
    convert(sys.argv[1], sys.argv[2])
