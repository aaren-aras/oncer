import os
import numpy as np
import cv2  # OpenCV
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Use GPU for faster training (if available)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print('Available GPUs:', physical_devices)
if physical_devices: 
  # Allocate memory incrementally, instead of all at once
  ts.config.experimental.set_memory_growth(physical_devices[0], True) 

# Paths (data from: https://www.kaggle.com/datasets/praneet0327/brain-tumor-dataset/data)
DATA_DIR = 'api/data/brain/training' 
MODEL_SAVE_PATH = 'api/models/brain_tumour_model.h5'

def load_data():
  categories = ['positive', 'negative']
  samples = []  # input: brain MRI scans (features, x)
  labels = []  # output: binary classification -> tumour detected (1) or not (0) (targets, y)

  for category in categories:
    folder = os.path.join(DATA_DIR, category)
    label = categories.index(category) 

    for img_name in os.listdir(folder):
      img_path = os.path.join(folder, img_name)
      img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
      img = cv2.resize(img, (128, 128))
      samples.append(img)
      labels.append(label)
  
  # Normalize image data
  samples = np.array(samples) / 255.0
  labels = np.array(labels)

  return samples, labels

# Convolutional Neural Network (CNN): https://www.youtube.com/watch?v=pj9-rr1wDhM
def build_model():
  model = Sequential()  # LINEAR stack of layers
  
  # (1) Convolutional layers (3x3 filters, -ve values -> 0 via ReLU, 150x150 img + 3 channels (RGB))
  model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))  # extract features
  # Reduce img size by 2x by taking max value in 2x2 grid (downsampling)
  model.add(MaxPooling2D(pool_size=(2, 2)))  # reduce complexity and avoid overfitting

  # "Abstraction": Increasingly complex feature extraction built off previous layers (32 -> 64 -> 128)
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(128, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  # (2) Fully connected layers ()
  model.add(Flatten())  # 2D -> 1D
  model.add(Dense(128, activation='relu'))  
  model.add(Dropout(0.5))
  model.add(Dense(2, activation='softmax'))  # output for 2 classes (yes/no)

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def train_model():
  data, labels = load_data()

  # One-hot encode labels
  labels = to_categorical(labels)

  # Split data into training and validation sets
  x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2)

  # Augment data
  datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, horizontal_flip=True)
  datagen.fit(x_train)

  model = build_model()

  model.save(MODEL_SAVE_PATH)
  print(f'Model saved to {MODEL_SAVE_PATH}')

if __name__ == '__main__':  # run script
  train_model()