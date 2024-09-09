import os
import random
import glob
import shutil

# Switch to working dir
os.chdir('../../data/brain')
SOURCE_DIR = os.getcwd()

# Define datasets
TRAIN_DIR = os.path.join(SOURCE_DIR, 'train')
VALID_DIR = os.path.join(SOURCE_DIR, 'valid')
TEST_DIR = os.path.join(SOURCE_DIR, 'test')

# Create 'positive' and 'negative' folders (if needed)
for dir_name in [TRAIN_DIR, VALID_DIR, TEST_DIR]:
  os.makedirs(os.path.join(dir_name, 'positive'), exist_ok=True)
  os.makedirs(os.path.join(dir_name, 'negative'), exist_ok=True)

# Move images back to source directory (if needed)
def move_back_to_src_dir(dir_name):
  for folder in ['positive', 'negative']:
    folder_path = os.path.join(dir_name, folder)
    for filename in os.listdir(folder_path):
      file_path = os.path.join(folder_path, filename)
      if os.path.isfile(file_path):
        shutil.move(file_path, os.path.join(SOURCE_DIR, filename))

move_back_to_src_dir(TRAIN_DIR)
move_back_to_src_dir(VALID_DIR)
move_back_to_src_dir(TEST_DIR)

# Randomly sample images from source directory based on category
def sample_images(category, num_samples):
  # E.g., ['api/data/brain/Tr-no_0840.jpg', 'api/data/brain/Tr-no_0812.jpg', ...] for category = 'no'
  return random.sample(glob.glob(os.path.join(SOURCE_DIR, f'*{category}*')), num_samples)

datasets = {
  'train': (500, 250),  # (neg, pos)
  'valid': (150, 75),  # 1/2 * pos since calling twice
  'test': (100, 50)
}

# Move images to respective directories based on sampling and distribution amounts
for dataset, (neg_samples, pos_samples) in datasets.items():
  neg_images = sample_images('no', neg_samples)
  
  # TO DO: organize based on tumour type (gliomas, meningiomas) if positive
  pos_images = sample_images('gl', pos_samples)
  pos_images += sample_images('me', pos_samples)

  for c in neg_images:  # c = current image
    try:
      # E.g., 'api/data/brain/Tr-no_0835.jpg' -> 'api/data/brain/train/negative/Tr-no_0835.jpg'
      shutil.move(c, os.path.join(dataset, 'negative'))
    except Exception as e:
      print(f'Error moving \'{os.path.basename(c)}\': {e}')
  
  for c in pos_images:
    try:
      # E.g., 'api/data/brain/Tr-gl_0336.jpg' -> 'api/data/brain/train/positive/Tr-gl_0336.jpg'
      shutil.move(c, os.path.join(dataset, 'positive'))
    except Exception as e:
      print(f'Error moving \'{os.path.basename(c)}\': {e}')

print('*COMPLETE: images have been distributed across training, validation, and testing sets')