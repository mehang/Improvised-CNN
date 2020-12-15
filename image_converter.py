import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os

from skimage.io import imread, imsave
from skimage.transform import resize
from skimage import color

from pathlib import Path

train_dir = "../dataset/train/"
test_dir = "../dataset/test1/"
print(os.listdir("../dataset"))


import cv2

def resize_images(input_dir, output_dir, size=(256,256)):
    filenames = os.listdir(input_dir)
    for filename in filenames:
        if 'cat' in filename:
            file_directory = output_dir + 'cat/'
        else:
            file_directory = output_dir + 'dog/'
        image_data = cv2.imread(input_dir + filename).astype(np.uint8)
        # resizing in float should it be converted to integer
        resized_image = cv2.resize(image_data, size)
        cv2.imwrite(file_directory+filename, resized_image)


resize_images(train_dir, "../dataset/cats-vs-dogs/train/")



