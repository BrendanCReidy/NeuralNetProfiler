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
print("------------------")
model = Sequential([
  Input(shape=(54,54,1)),
  Conv2D(6, kernel_size=(5,5), activation="tanh", padding='same'),
  AveragePooling2D(pool_size=(2,2), strides=2, padding='valid'),
  Conv2D(16, kernel_size=(5,5), activation="tanh", padding='valid'),
  AveragePooling2D(pool_size=(2,2), strides=2, padding='valid'),
  Conv2D(120, kernel_size=(5,5), activation="tanh", padding='valid'),
  Flatten(),
  Dense(84, activation="tanh"),
  Dense(3,activation="softmax"),
])

features = np.random.rand(1,54,54,1)
flagLayer = 0

l = model.layers[flagLayer]

intermediate_model = Model(inputs=model.layers[0].input,outputs=l.output)

print("Flagging at: " + l.name)
intermediate_model.predict(features) # This is the only line responsible for inference