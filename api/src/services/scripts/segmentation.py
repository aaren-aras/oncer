import io
import base64

import numpy as np
from PIL import Image

from src.services.config import BASE_COLOR, ALPHA_NCR, ALPHA_ED, ALPHA_ET, LABEL_MAP


def create_overlay(mask: np.ndarray) -> str:
    """
    Converts a segmentation mask into a base64-encoded RGBA overlay image for UI display.
     - Input shape: (H, W) from 'model.py'
     - Output shape: (H, W, 4) where 4 = 4 RGBA channels
    """
    
    # Define overlay colours for BrATS tumour regions with varying opacities  
    color_map = {
        0: (0, 0, 0, 0), # Background
        1: (*BASE_COLOR, ALPHA_NCR), # Necrotic tumour core (NCR) 
        2: (*BASE_COLOR, ALPHA_ED), # Peritumoural edematous/invaded tissue (ED)
        3: (*BASE_COLOR, ALPHA_ET), # Gadolinium-enhancing tumour (ET)
    }

    # Initialize RGBA img arr with same dims as input mask (IMG_SIZE in 'config.py')
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    
    # Paint overlay pixels with config colours per BraTS labels
    for label, color in color_map.items():
        overlay[mask == label] = color 

    img = Image.fromarray(overlay, mode='RGBA') # np arr -> PIL Image obj
    buffer = io.BytesIO()
    img.save(buffer, format='PNG') # PIL image obj -> in-memory file-like obj as PNG (supports transparency)
    # return base64.b64encode(buffer.getvalue()).decode('utf-8') # PNG -> bytes -> base64 str  
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


def compute_segmentation_stats(mask: np.ndarray) -> dict[str, object]:
    """
    From a predicted segmentation mask (H, W) with labels 0-3, compute:
    - Class pixel counts,
    - % coverage (excluding background).
    """
    height, width = mask.shape 
    total_pixels = height * width

    class_names = list(LABEL_MAP.values())
    class_counts = { class_names[i]: int((mask == i).sum()) for i in range(4) }

    non_bkgd_pixels = max(total_pixels - class_counts['background'], 1) # avoid division-by-0 cases (i.e., empty mask)

    coverage_percent = {
        # Calculate % of non-background area each tumour region takes up
        n: round((c / non_bkgd_pixels) * 100, 2)
        for n, c in class_counts.items()
        if n != 'background'
    }

    return {
        'dimensions': [height, width],
        'labels': class_counts,
        'coverage_percent': coverage_percent
    }
