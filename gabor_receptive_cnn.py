import numpy as np

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from itertools import product

import cv2

def get_gabor_filters(filters_num, channels):
    lambds = [np.pi/4, np.pi/3, np.pi/2, np.pi]
    total_thetas = filters_num//len(lambds)
    thetas = [np.pi * (i / total_thetas) for i in range(total_thetas)]
    sigmas = [3]
    gammas = [0.4]
    gabor_params = list(product(sigmas, lambds, thetas, gammas))
    filters = []
    for sigma, lambd, theta, gamma in gabor_params:
        filter = cv2.getGaborKernel(ksize=(3,3), sigma=sigma, theta=theta, lambd=lambd, gamma=gamma)
        filters.append(np.repeat([filter], channels, axis=0))
    return filters


def display_filters(filters):
    for i in range(64):
        cv2.imshow("Gabor Kernel", cv2.resize(filters[i][0],(400,400)))
        cv2.waitKey()
    cv2.destroyAllWindows()


receptive_filters = get_gabor_filters(64,3)

# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train = x_train.astype('float32')/255
# x_test = x_test.astype('float32')/255
#
# y_train = to_categorical(y_train, num_classes=10)
# y_test = to_categorical(y_test, num_classes=10)

input_dim = (32,32,3)
input_img = Input(shape=input_dim)

cl1_template = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same',
             input_shape=input_dim, activation='relu')
cl1 = cl1_template(input_img)

receptive_weights = cl1_template.get_weights()
receptive_weights[0] = receptive_filters
cl1_template.set_weights(receptive_weights)


pl1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(cl1)



