
import seaborn as sns
import os, shutil

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Model
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import GlobalMaxPooling2D
tf.__version__
import cv2
import pytesseract
from PIL import Image
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import swifter
from IPython.display import Image
import pydotplus
import pickle
from sklearn.metrics.pairwise import pairwise_distances
from keras import models

import preprocess as pre


def load_image(df, index, images_folder_path):
    return cv2.imread(images_folder_path+df.filename[index])


def get_embedding(model, image_filename, images_folder_path):
    """
    Takes .... then reshapes and converts it into an array
    """

    img = image.load_img(images_folder_path + image_filename,
                         target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x).reshape(-1)
