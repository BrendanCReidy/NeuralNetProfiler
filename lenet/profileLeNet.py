import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import AveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras import Input, Model, Sequential
from keras.utils import to_categorical
from sklearn.metrics import precision_recall_fscore_support
tf.get_logger().setLevel('ERROR')
import sys
class FlagLayer(keras.layers.Layer):
    def __init__(self):
        super(FlagLayer, self).__init__()

    def call(self, inputs):
        print("Layer called")
        return inputs

model = Sequential([
  Conv2D(filters=6, kernel_size=(5, 5), activation='tanh', input_shape=(32,32,1)),
  AveragePooling2D(),
  Conv2D(filters=16, kernel_size=(5, 5), activation='tanh'),
  AveragePooling2D(),
  Activation('tanh'),
  Flatten(),
  Dense(units=120, activation='tanh'),
  Dense(units=84, activation='tanh'),
  Dense(units=10, activation = 'sigmoid'),
  FlagLayer(),
])

features = np.ones((1,32,32,1))*4
model.predict(features)