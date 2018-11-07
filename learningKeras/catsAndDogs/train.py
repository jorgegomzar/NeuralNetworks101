import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import time

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

NAME = "Cats-vs-dog-cnn-64x2-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)  # 1/3 of process capacity
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Loading the training data from the pickles
pickle_in = open('X.pickle', 'rb')
X = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('Y.pickle', 'rb')
Y = pickle.load(pickle_in)
pickle_in.close()

X = X / 255.0

dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

model = Sequential()
# 1st layer
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:])) # 64 neurons Convolution layer, window = 3x3, with input_shape of the x shape
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# 2nd layer
model.add(Conv2D(64, (3,3))) # 64 neurons Convolution layer, window = 3x3
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# 3rd layer
model.add(Flatten()) # Converts 3D feature maps to 1D feature vectors

model.add(Dense(64))
model.add(Activation('relu'))

# Output layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
model.fit(X, Y, batch_size=32, validation_split=0.1, epochs = 10, callbacks=[tensorboard])

model.save('animal_classifier-v{}.model'.format('1'))
