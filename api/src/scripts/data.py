from pathlib import Path
import json

from tqdm import tqdm
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split

from ..config import MODALITIES, EPSILON, LABEL_MAP

SCRIPT_DIR = Path(__file__).resolve().parent # oncer/api/src/scripts
BRATS_DIR = (SCRIPT_DIR / '../../../data/BraTS2021_Training_Data').resolve() # update if needed
OUTPUT_DIR = (SCRIPT_DIR / '../../../data/BraTS2021_Processed_Data').resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define dataset subdirectories    
IMG_DIR = OUTPUT_DIR / 'images'
MASK_DIR = OUTPUT_DIR / 'masks'
METADATA_DIR = OUTPUT_DIR / 'metadata'

# Create above subdirectories and 'train', 'valid', and 'test' subsubdirectories
for split in ['train', 'valid', 'test']:
    (IMG_DIR / split).mkdir(parents=True, exist_ok=True)
    (MASK_DIR / split).mkdir(parents=True, exist_ok=True)
    (METADATA_DIR / split).mkdir(parents=True, exist_ok=True)


def normalize_modality(img: np.ndarray) -> np.ndarray:
    """
    Normalize MRI image slice to 8-bit greyscale to standardize pixel intensities and highlight structure over brightness.
    """
    img = np.nan_to_num(img) # handle any corrupted/missing values
    img = np.clip(img, 0, np.percentile(img, 99)) # cut off extremes (99th %tile)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + EPSILON) # scale to [0, 1]
    return (img * 255).astype(np.uint8) # scale to [0, 255] (2e8)


def process_subject(subject_path: Path) -> list[tuple[np.ndarray, np.ndarray, str, int]]:
    """
    Given a BraTS subject (BraTS_2021_0xxxx):
    - Stack all 4 MRI modalities (T1, T1CE, T2, FLAIR) into a 4-channel 3D image volume,
    - Extract MRI image slices from both the image volume and corresponding 3D segmentation mask.
    """
    subject_id = subject_path.name
    
    imgs = []
    for m in MODALITIES:
        m_nii = nib.load(subject_path / f'{subject_id}_{m}.nii.gz')
        data = normalize_modality(m_nii.get_fdata()) # Nifti1Image obj -> np arr (raw voxels as float) -> np arr (raw voxels normalized to [0, 255])
        imgs.append(data)
    stacked = np.stack(imgs, axis=-1) # shape (H, W, D) x 4 -> (H, W, D, 4) (append new dim at the end)

    seg_nii = nib.load(subject_path / f'{subject_id}_seg.nii.gz')
    mask = seg_nii.get_fdata().astype(np.uint8) # Nifti1Image obj -> np arr (raw voxels as float) -> np arr (raw voxels as uint8)  
    mask[mask == 4] = 3 # remap label 4 -> 3 for convenience

    slices = []
    for i in range(stacked.shape[2]): # per axial slice (z-resolution)
        img_slice = stacked[:, :, i, :] # shape (H, W, 4)
        mask_slice = mask[:, :, i] # shape (H, W)
        slices.append(img_slice, mask_slice, subject_id, i)

    return slices


def save_slice(img_stack: np.ndarray, mask_slice: np.ndarray, subject_id: str, slice_idx: int, split: str) -> None:
    """
    Save MRI image slice, corresponding segmentation mask, and per-slice metadata to their respective subsubdirectories.
    """
    img_path = IMG_DIR / split / f'{subject_id}_slice{slice_idx:03d}.npy' # e.g., 5 -> 005
    mask_path = MASK_DIR / split / f'{subject_id}_slice{slice_idx:03d}_mask.npy'
    metadata_path = METADATA_DIR / split / f'{subject_id}_slice{slice_idx:03d}.json'

    np.save(img_path, img_stack) # 4-channel (T1, T1CE, T2, FLAIR): shape (H, W, 4), uint8 (0-255 greyscale)
    np.save(mask_path, mask_slice) # 1-channel (class labels): shape (H, W), uint8 (0: background, 1: NCR, 2: ED, 4->3: ET)
    
    total_pixels = mask_slice.size # total = H * W
    unique, counts = np.unique(mask_slice, return_counts=True) # num pixels belonging to each class
    class_pixel_counts = {int(u): int(c) for u, c in zip(unique, counts)}

    # Append per-slice stats to metadata JSON
    stats = {}
    for label_id, label_name in LABEL_MAP.items():
        count = class_pixel_counts.get(label_id, 0)
        stats[label_name] = {
            'pixel_count': count,
            'percent': round((count / total_pixels * 100), 2)
        }
    
    metadata = {
        'subject_id': subject_id,
        'segmentation_stats': stats
    }

    with open(metadata_path, 'w') as file:
        json.dump(metadata, file, default=str) # ->str if unserializable


def prepare_data() -> None:
    """
    Process all BraTS subjects and prepare dataset for model training.
    """
    all_slices = []
    
    subjects = sorted([p for p in BRATS_DIR.iterdir() if p.is_dir()])
    for subject_path in tqdm(subjects, desc='Processing BraTS2021 subjects'):
        slices = process_subject(subject_path)
        all_slices.extend(slices)
    print(f'Total 2D Slices: {len(all_slices)}')
    
    # Split MRI image slices into training, validation, and test sets (70-15-15)
    train, temp = train_test_split(all_slices, test_size=0.3, random_state=2025) # 70% train, 30% temp
    valid, test = train_test_split(temp, test_size=0.5, random_state=2025) # of 30% temp: 50% valid, 50% test

    splits = [(train, 'train'), (valid, 'valid'), (test, 'test')]
    for split_data, split_name in splits:
        for img, mask, subject_id, idx in tqdm(split_data, desc=f'Saving \'{split_name}\' slices'):
            save_slice(img, mask, subject_id, idx, split_name)

    print('*COMPLETE: images have been distributed across training, validation, and test sets')


if __name__ == '__main__':
    prepare_data()
