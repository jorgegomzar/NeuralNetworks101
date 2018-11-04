import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Loading the training data from the pickles
pickle_in = open('X.pickle', 'rb')
X = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open('Y.pickle', 'rb')
Y = pickle.load(pickle_in)
pickle_in.close()

X = X / 255.0
# Input layer
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
model.add(Flatten())
model.add(Dense(64))
# Output layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
model.fit(X, Y, batch_size=32, validation_split=0.1, epochs = 10)

model.save('animal_classifier.model')
