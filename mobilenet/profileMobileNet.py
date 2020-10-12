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
tf.get_logger().setLevel('ERROR')
print("------------------")
mobile = tf.keras.applications.mobilenet.MobileNet()


features = np.random.rand(1,224,224,3)
count = 1


flagLayer = 1
for l in mobile.layers[1:]:
    intermediate_model = Model(inputs=mobile.layers[0].input,outputs=l.output)
    if(count==flagLayer):
        print("Flagging at: " + l.name)
        intermediate_model.predict(features) # This is the only line responsible for inference
    count=count+1