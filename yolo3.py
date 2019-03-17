# Keras functions
# Try these CNN later on
#from keras.applications import ResNet50
#from keras.applications import InceptionV3
#from keras.applications import Xception # TensorFlow ONLY
#from keras.applications import VGG16
#from keras.applications import VGG19
#from keras.applications import imagenet_utils
#from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import load_model, Model
from keras.layers import Input, Lambda, Conv2D
from keras import backend as K
import tensorflow as tf
import numpy as np
import argparse
import sys
# YOLOv3 model functions
# sys.path.append('/mnt/python/keras-yolo3/')
from yolo3.model import yolo_eval # Evaluate YOLO model on given input and return filtered boxes.
# from pydarknet import Detector, Image # YOLOv3 package
# import cv2 # OpenCV, OpenCV 3.4.1 will fail with darknet 
import yolo
import yolo_video


# yolo_model = load_model("model_data/yolo-openimages.h5") # load the model
# yolo_model.summary() # show a summary of the model layers
# configure the default to YOLOv3 on Open Images
yolo.YOLO._defaults['model_path']='model_data/yolo-openimages.h5'
yolo.YOLO._defaults['classes_path']='model_data/openimages.names'
yolo.YOLO._defaults['anchors_path']='model_data/yolo_anchors.txt'
yolo_video.detect_img(yolo.YOLO()) # comment r_image.show(), and add r_image.save(filename) to yolo_video.py
# yolo_video.detect_img() uses yolo.detect_image(), with additional ability to input multiple images on the fly