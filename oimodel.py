import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow


import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

def yoloModel (input_shape = (64,64,3), classes=601):

    X_input = Input(input_shape)

    X = Conv2D(filters=32, kernel_size=3, strides = 1, padding = 1, activation = 'leaky', batch_normalize =1)
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

model = yoloModel()
print(model.summary())