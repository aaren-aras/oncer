import os
import random
import glob
import shutil

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Switch to working dir (data from: https://www.kaggle.com/datasets/praneet0327/brain-tumor-dataset/data)
os.chdir('../../data/brain')
SOURCE_DIR = os.getcwd()

# Define dataset paths
TRAIN_DIR = os.path.join(SOURCE_DIR, 'train')
VALID_DIR = os.path.join(SOURCE_DIR, 'valid')
TEST_DIR = os.path.join(SOURCE_DIR, 'test')

# Create 'negative' and 'positive' folders (if needed)
for dir_name in [TRAIN_DIR, VALID_DIR, TEST_DIR]:
  os.makedirs(os.path.join(dir_name, 'negative'), exist_ok=True)
  os.makedirs(os.path.join(dir_name, 'positive'), exist_ok=True)

# Move images back to source directory (if needed)
def move_back_to_src_dir(dir_name):
  for folder in ['negative', 'positive']:
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

'''
TO DO:
 - Refactor and use augmentation (e.g., `rotation_range=10, zoom_range=0.1, horizontal_flip=True`)?
 - Use VGG16 (`preprocessing_function=tf.keras.applications.vgg16.preprocess_input`), then duplicate 
 channels for RGB (https://tinyurl.com/yzxkhmdh)? Not ideal?
'''

# Preprocess image data before inputting into CNN 
TRAIN_BATCHES = ImageDataGenerator(rescale=1./255) \
  .flow_from_directory(directory=TRAIN_DIR, target_size=(224, 224), classes=['negative', 'positive'], batch_size=10)  # `rescale`: normalize pixel values to [0, 1]

VALID_BATCHES = ImageDataGenerator(rescale=1./255) \
  .flow_from_directory(directory=VALID_DIR, target_size=(224, 224), classes=['negative', 'positive'], batch_size=10)  # `batch_size`: learns from 10 images at a time, then updates weights

TEST_BATCHES = ImageDataGenerator(rescale=1./255) \
  .flow_from_directory(directory=TEST_DIR, target_size=(224, 224), classes=['negative', 'positive'], batch_size=10, shuffle=False)  # `shuffle`: maintain correct order for confusion matrix

# Ensure console output is as expected
assert TRAIN_BATCHES.n == 1000 
assert VALID_BATCHES.n == 300 
assert TEST_BATCHES.n == 200 
assert TRAIN_BATCHES.num_classes == VALID_BATCHES.num_classes == TEST_BATCHES.num_classes == 2

samples, labels = next(TRAIN_BATCHES)
def plotImages(img_arr):
  fig, axes = plt.subplots(1, 10, figsize=(20, 20))
  axes = axes.flatten()  # 2D -> 1D
  for img, axis in zip(img_arr, axes):  # iterate over two lists in parallel
    axis.imshow(img)
    axis.axis('off')
  plt.tight_layout()
  plt.show()

print(labels)  # one-hot encoded (avoid implying ranking/order): [1. 0.] = neg, [0. 1.] = pos (floats!)
plotImages(samples)
