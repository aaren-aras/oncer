import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

def load_nifti(path):
    nii = nib.load(path)
    return nii.get_fdata()

# Adjust to your local path
os.chdir('../../data/BraTS2020/BraTS20_Training_001')
# SOURCE_DIR = os.getcwd()

# case_path = "brats2020/images/BraTS20_Training_001"
case_path = os.getcwd()
flair = load_nifti(os.path.join(case_path, "BraTS20_Training_001_flair.nii"))
seg = load_nifti(os.path.join(case_path, "BraTS20_Training_001_seg.nii"))

# Show a middle slice
slice_idx = flair.shape[2] // 2
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(flair[:, :, slice_idx], cmap='gray')
plt.title("FLAIR MRI")

plt.subplot(1, 2, 2)
plt.imshow(flair[:, :, slice_idx], cmap='gray')
plt.imshow(seg[:, :, slice_idx], alpha=0.3)  # Overlay segmentation
plt.title("FLAIR + Segmentation")

plt.tight_layout()
plt.show()
