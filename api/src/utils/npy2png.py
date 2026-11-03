import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def normalize(slice):
    """Normalize slice to [0,255] uint8."""
    slice = slice.astype(np.float32)
    slice = (slice - np.min(slice)) / (np.max(slice) - np.min(slice) + 1e-6)
    return (slice * 255).astype(np.uint8)

def npy_to_png(input_path, output_dir, base_name=None, colorize_mask=True):
    """
    Converts a 4-channel .npy image or 2-channel .npy mask to:
      - RGB composite PNG (for preview),
      - individual grayscale PNGs for each modality.
    """
    os.makedirs(output_dir, exist_ok=True)

    arr = np.load(input_path)  # shape: (H, W, 4)
    # if arr.shape[-1] != 4:
    #     raise ValueError(f"Expected 4-channel input, got shape: {arr.shape}")

    if base_name is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]


    if arr.ndim ==3 and arr.shape[-1] == 4:
        # Normalize each modality
        channels = [normalize(arr[:, :, i]) for i in range(4)]

        # Save each channel as grayscale PNG
        modalities = ['FLAIR', 'T1', 'T1CE', 'T2']
        for i, (modality, channel_img) in enumerate(zip(modalities, channels)):
            Image.fromarray(channel_img).save(os.path.join(output_dir, f"{base_name}_{modality}.png"))

        # Create RGB composite using:
        # R = T1CE (tumor enhancing)
        # G = FLAIR (edema)
        # B = T2 (fluid)
        rgb = np.stack([channels[2], channels[0], channels[3]], axis=-1)  # (H, W, 3)
        Image.fromarray(rgb).save(os.path.join(output_dir, f"{base_name}_composite.png"))

        print(f"[✓] Saved preview to: {output_dir}")
    
    elif arr.ndim == 2:
          # ---- 2D mask ----
        Image.fromarray(arr.astype(np.uint8)).save(
            os.path.join(output_dir, f"{base_name}_mask.png")
        )
        print(f"[✓] Saved 2D grayscale mask to: {output_dir}")

        if colorize_mask:
            # Save a colorized version using matplotlib
            plt.imsave(
                os.path.join(output_dir, f"{base_name}_mask_colored.png"),
                arr,
                cmap='nipy_spectral',
                vmin=0, vmax=3
            )
            print(f"[✓] Saved colorized mask to: {output_dir}")

    else:
        raise ValueError(f"Unsupported input shape: {arr.shape}")
        

if __name__ == "__main__":
    # Example usage
    npy_to_png(
        input_path="../../data/BraTS2021_Processed_Data/masks/test/BraTS2021_01664_slice103_mask.npy",
        # input_path="../../data/BraTS2021_Processed_Data/images/test/BraTS2021_01666_slice105.npy",
        output_dir="./previews"
    )
