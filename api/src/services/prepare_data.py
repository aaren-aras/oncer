import os
import random
import glob
import shutil

SOURCE_DIR = 'api/data/brain'
TRAIN_DIR = os.path.join(SOURCE_DIR, 'train')
VALID_DIR = os.path.join(SOURCE_DIR, 'valid')
TEST_DIR = os.path.join(SOURCE_DIR, 'test')

# Create 'positive' and 'negative' folders (for each) if they do not exist
for dir_name in [TRAIN_DIR, VALID_DIR, TEST_DIR]:
  os.makedirs(os.path.join(dir_name, 'positive'), exist_ok=True)
  os.makedirs(os.path.join(dir_name, 'negative'), exist_ok=True)

# Clear existing images in specified directories
def clear_directory(directory):
  for folder in ['positive', 'negative']:
    folder_path = os.path.join(directory, folder)
    for filename in os.listdir(folder_path):
      file_path = os.path.join(folder_path, filename)
      if os.path.isfile(file_path):
        os.remove(file_path)

clear_directory(TRAIN_DIR)
clear_directory(VALID_DIR)
clear_directory(TEST_DIR)

# Sample images from source directory based on category
def sample_images(category, num_samples):
  # E.g., ['api/data/brain/Tr-no_0840.jpg', 'api/data/brain/Tr-no_0812.jpg', ...] for category = 'no'
  return random.sample(glob.glob(os.path.join(SOURCE_DIR, f'*{category}*')), num_samples)

datasets = {
  'train': (500, 500),  # (pos, neg)
  'valid': (150, 150),
  'test': (100, 100)
}

for dataset, (neg_samples, pos_samples) in datasets.items():
  neg_images = sample_images('no', neg_samples)
  # TO DO: organize based on tumour type (gliomas, meningiomas, ...) if positive
  pos_images = sample_images('', pos_samples)
  
  for c in neg_images:  # c = current image
    # E.g., 'api/data/brain/Tr-no_0835.jpg' -> 'api/data/brain/train/negative/Tr-no_0835.jpg'
    shutil.move(c, os.path.join(SOURCE_DIR, dataset, 'negative', os.path.basename(c)))
  for c in pos_images:
    shutil.move(c, os.path.join(SOURCE_DIR, dataset, 'positive', os.path.basename(c)))

print('Images moved to training, validation, and testing sets successfully!')