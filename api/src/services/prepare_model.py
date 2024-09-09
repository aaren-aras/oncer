import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.metrics import confusion_matrix

from prepare_data import TRAIN_BATCHES, VALID_BATCHES

# Use GPU for faster training (if available)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print('Available GPUs:', physical_devices)
if physical_devices: 
  # Allocate memory incrementally, instead of all at once
  ts.config.experimental.set_memory_growth(physical_devices[0], True) 

# Convolutional Neural Network (CNN): https://www.youtube.com/watch?v=pj9-rr1wDhM
def build_model():
  model = Sequential()  # LINEAR stack of layers
  
  # (1) Convolutional layers (3x3 filters, -ve values -> 0 via ReLU, 150x150 img + 1 channel (greyscale))
  model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 1))) 
  # Reduce img size by 2x by taking max value in 2x2 grid (downsampling)
  model.add(MaxPool2D(pool_size=(2, 2)))  # reduce complexity and avoid overfitting

  # "Abstraction": increasingly complex feature extraction built off prev. layers (32 -> 64 -> 128 filters)
  model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
  model.add(MaxPool2D(pool_size=(2, 2)))

  model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
  model.add(MaxPool2D(pool_size=(2, 2)))

  # (2) Fully connected layers (every input neuron connected to every output neuron)
  model.add(Flatten())  # 2D -> 1D
  model.add(Dense(units=128, activation='relu'))  
  model.add(Dropout(0.5))  # randomly set 50% of neurons to 0 to prevent overfitting
  '''TO DO: use `binary_crossentropy`, sigmoid, and `units=1` instead (for binary classification)'''
  # Outputs probabilities for 2 categories (0 -> neg, 1 -> pos): e.g., [0.7, 0.3] -> negative
  model.add(Dense(units=2, activation='softmax')) 

  model.summary()
  # Optimze weights while minimizing loss
  model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def train_model():
  model = build_model()
  '''TO DO: resolve runtime warnings'''
  # Epochs: num. of times the model will cycle through data, verbose: amount of console output
  model.fit(x=TRAIN_BATCHES, validation_data=VALID_BATCHES, epochs=10, verbose=2)
  model.save('../models/brain_tumour_model.keras')
  print(f'*COMPLETE: model trained and saved to \'models\' folder')

if __name__ == '__main__':  # run script
  train_model()