import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, UpSampling2D, Concatenate
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
from keras.layers import LeakyReLU
import tensorflow as tf
import json
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
from keras.utils import plot_model


def get_Parameters():
    data = {}
    keys = []
    li = []
    with open('conv_config.json') as json_file:  
        data = json.load(json_file)
    for key in data.keys():
        keys.append(int(key))
    keys.sort()
    for k in keys:
        li.append(data[str(k)])
    return li

def set_weights(model):
    pre_trained_model = load_model("yolo-openimages.h5")
    for i in range(0,252):
        print(i)
        model.layers[i].set_weights(pre_trained_model.layers[i].get_weights())
    return model

def conv_layer(X, params, ctr):
    return Conv2D(filters = params[ctr]["filters"], kernel_size=params[ctr]["kernel_size"], strides=params[ctr]["strides"], padding=params[ctr]["padding"], use_bias=params[ctr]["use_bias"], name = "conv2d_"+str(ctr+1))(X), (ctr+1)

def batch_norm(X, b_ctr):
    return BatchNormalization(axis = 3, name = 'bn_conv'+str(b_ctr))(X), (b_ctr+1)

def conv_block(X, params, ctr, b_ctr):
    X, ctr = conv_layer(X,params,ctr)
    X, b_ctr = batch_norm(X, b_ctr)
    X = LeakyReLU()(X)
    return X, (ctr), (b_ctr)

def yoloModel (input_shape = (608,608,3), classes=601):

    params = get_Parameters()
    ctr = 0
    b_ctr = 1
    X_input = Input(input_shape)
    #using initializer here
    # X = Conv2D(filters=32, kernel_size=3, strides = 1, padding = 'same', kernel_initializer=Variance)(X_input)
    
    X, ctr, b_ctr = conv_block(X_input, params, ctr, b_ctr)
    X = ZeroPadding2D()(X)
    X_1, ctr, b_ctr = conv_block(X, params, ctr, b_ctr) #leaky relu 2
    X, ctr, b_ctr = conv_block(X_1, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X = Add()([X, X_1]) #add_1
    X = ZeroPadding2D()(X)
    X_2, ctr, b_ctr = conv_block(X, params, ctr, b_ctr) #leaky_relu 5
    X, ctr, b_ctr = conv_block(X_2, params, ctr, b_ctr)#leaky relu 6
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X_3 = Add()([X, X_2])
    X, ctr, b_ctr = conv_block(X_3, params, ctr, b_ctr)#leaky relu 6
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X = Add()([X,X_3])
    X = ZeroPadding2D()(X)
    X_4, ctr, b_ctr = conv_block(X, params, ctr, b_ctr) #leakyRelu 10
    X, ctr, b_ctr = conv_block(X_4, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X_5 = Add()([X,X_4])
    X, ctr, b_ctr = conv_block(X_5, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr) #leakyRelu 14
    X_6 = Add()([X,X_5])
    X, ctr, b_ctr = conv_block(X_6, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X_7 = Add()([X,X_6])
    X, ctr, b_ctr = conv_block(X_7, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X_8 = Add()([X,X_7])
    X, ctr, b_ctr = conv_block(X_8, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X_9 = Add()([X, X_8])
    X, ctr, b_ctr = conv_block(X_9, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X_10 = Add()([X,X_9])
    X, ctr, b_ctr = conv_block(X_10, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr) #leakyRelu 24
    X_11 = Add()([X,X_10])
    X, ctr, b_ctr = conv_block(X_11, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X_12 = Add()([X,X_11])#yet to add
    X = ZeroPadding2D()(X_12)
    X_13, ctr, b_ctr = conv_block(X, params, ctr, b_ctr) #leakyRelu 27
    X, ctr, b_ctr = conv_block(X_13, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X_14 = Add()([X,X_13])
    X, ctr, b_ctr = conv_block(X_14, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X_15 = Add()([X, X_14])
    X, ctr, b_ctr = conv_block(X_15, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr) #leakyRelu 33
    X_16 = Add()([X, X_15])
    X, ctr, b_ctr = conv_block(X_16, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X_17 = Add()([X, X_16])
    X, ctr, b_ctr = conv_block(X_17, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X_18 = Add()([X, X_17])
    X, ctr, b_ctr = conv_block(X_18, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X_19 = Add()([X, X_18])
    X, ctr, b_ctr = conv_block(X_19, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X_20 = Add()([X,X_19])
    X, ctr, b_ctr = conv_block(X_20, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X_21 = Add()([X,X_20])#yet to add
    X = ZeroPadding2D()(X_21)
    X_22, ctr, b_ctr = conv_block(X, params, ctr, b_ctr) #leakyRelu 44
    X, ctr, b_ctr = conv_block(X_22, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X_23 = Add()([X, X_22])
    X, ctr, b_ctr = conv_block(X_23, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X_24 = Add()([X, X_23])
    X, ctr, b_ctr = conv_block(X_24, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr) #leakyRelu 50
    X_25 = Add()([X, X_24])
    X, ctr, b_ctr = conv_block(X_25, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X = Add()([X, X_25])
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr) #leakRelu 56
    X_26, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)#leakyRelu 57

    # conv_58, ctr, b_ctr = conv_block(X_26, params, ctr, b_ctr)
    # conv_59, ctr = conv_layer(conv_58, params, ctr) #conv_59

    X, ctr, b_ctr = conv_block(X_26, params, ctr, b_ctr)#leakyRelu59
    X = UpSampling2D(size=(2, 2))(X)
    X = Concatenate()([X,X_21])
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X_27, ctr, b_ctr = conv_block(X, params, ctr, b_ctr) #leakyRelu64

    # X, ctr, b_ctr = conv_block(X_27, params, ctr, b_ctr)
    # X, ctr = conv_layer(X, params, ctr) #conv_67

    X, ctr, b_ctr = conv_block(X_27, params, ctr, b_ctr)
    X = UpSampling2D(size = (2,2))(X)
    X = Concatenate()([X,X_12])
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr) #leakyRelu 67
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr) #leakyRelu 69
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr)
    X, ctr, b_ctr = conv_block(X, params, ctr, b_ctr) #leakyRelu 71
    Conv_58 , ctr = conv_layer(X_26, params, ctr)
    Conv_66 , ctr = conv_layer(X_27, params, ctr)
    Conv_74 , ctr = conv_layer(X, params, ctr)
    Conv_58, b_ctr = batch_norm(Conv_58, b_ctr)
    Conv_66, b_ctr = batch_norm(Conv_66, b_ctr)
    Conv_74, b_ctr = batch_norm(Conv_74, b_ctr)
    Conv_58 = LeakyReLU()(Conv_58)
    Conv_66 = LeakyReLU()(Conv_66)
    Conv_74 = LeakyReLU()(Conv_74)
    Conv_59= Conv2D(filters = 1818, kernel_size = (1,1), strides = (1,1), padding = 'same', use_bias=True, activation = 'linear', name = "conv2d_"+str(ctr+1))(Conv_58)
    Conv_67 = Conv2D(filters = 1818, kernel_size = (1,1), strides = (1,1), padding = 'same', use_bias=True, activation = 'linear', name = "conv2d_"+str(ctr+2))(Conv_66)
    Conv_75 = Conv2D(filters = 1818, kernel_size = (1,1), strides = (1,1), padding = 'same', use_bias=True, activation = 'linear', name = "conv2d_"+str(ctr+3))(Conv_74)
    model = Model(inputs = X_input, outputs = [Conv_59,Conv_67,Conv_75], name='yolo_v3_open_images')
    
    return model

model = yoloModel()
# plot_model(model, to_file='model.png', show_shapes=True,show_layer_names=True) #use it to plot the graph in a png
model.compile(optimizer='adam', loss='mean_squared_error')

print(len(model.output)) #should be 3
# model = set_weights(model) #Comment out this line to use pjreddies weight

print(model.summary())
model.save('yolo-openimages-custom.h5')