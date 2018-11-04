# First I am going to load the dataset and process it so my
# neural network can work with it

import os
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATADIR = 'C:/Datasets/petImages'
CATEGORIES = ['Dog', 'Cat']
IMG_SIZE = 100
training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # path to cats or dogs dir
        class_num = CATEGORIES.index(category) # To label the data
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) # Load image as grayscale
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) # Resize image so every picture has the same size
                training_data.append([new_array, class_num]) # Append to the training data array
            except Exception as e:
                pass

create_training_data() # Run the function
print(len(training_data)) # Debugging purposes
random.shuffle(training_data) # Mix everything up

X, Y = [], [] # Initialize arrays

for features, label in training_data: # Saving data to X(input) or Y(output expected)
    X.append(features)
    Y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # Resizing our input data, 1 stands for grayscale, 3 would be color

# Now I save the training_data to use it in the neural network's training
pickle_out = open('X.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('Y.pickle', 'wb')
pickle.dump(Y, pickle_out)
pickle_out.close()
