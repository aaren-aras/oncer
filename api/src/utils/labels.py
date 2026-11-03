import os

import nibabel as nib
import numpy as np

CURRENT_DIR = '../../data/BraTS2021_Training_Data/BraTS2021_00000/BraTS2021_00000_seg.nii.gz'
mask_nii = nib.load(CURRENT_DIR)
mask_data = mask_nii.get_fdata()
unique_labels = np.unique(mask_data)
print(f'Unique labels in mask: {unique_labels}')
