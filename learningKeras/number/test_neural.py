import tensorflow as tf
import keras
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Para que no muestre basura

mnist = keras.datasets.mnist # 28x28 images of hand-written digits 0-9

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)

# --------- TEST ----------
new_model = keras.models.load_model('epic_num_reader.model')

predictions = new_model.predict([x_test])
for i in range(20):
    plt.imshow(x_test[i+10], cmap = plt.cm.binary)
    plt.show()
    print(np.argmax(predictions[i+10]))
