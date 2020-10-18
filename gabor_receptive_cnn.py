import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from sklearn.model_selection import train_test_split

from itertools import product
import math

import cv2
import random


def calc_gabor(x,y,theta, lambd, sigma, psi, gamma):
    x_rot = x*math.cos(theta) + y*math.sin(theta)
    y_rot = -x*math.sin(theta) + y*math.cos(theta)
    sigma_x = sigma
    sigma_y = sigma/gamma

    g = (1/(2*math.pi*sigma_x*sigma_y))*math.exp((x_rot/sigma_x)**2+(y_rot/sigma_y)**2) * math.cos(2*math.pi*lambd + psi)
    return g


def get_gabor_filters(filters_num, channels):
    '''
    Creates gabor filters of size 3*3. The gabor filter is repeated for
    all the given number of channels, but it changes across according to
    the number of filters needed.
    :param filters_num: The number of filters needed - gabor filter is changed
    across it
    :param channels: The number of channels - gabor filter is duplicated for it
    :return: Array of gabor filters, dimension is (filters_num,channels, 3, 3)
    '''
    lambds = [np.pi/4, np.pi/3, np.pi/2, np.pi]
    total_thetas = filters_num//len(lambds)
    thetas = [np.pi * (i / total_thetas) for i in range(total_thetas)]
    sigmas = [10]
    gammas = [0.25]
    gabor_params = list(product(sigmas, lambds, thetas, gammas))
    filters = []
    for sigma, lambd, theta, gamma in gabor_params:
        sigma = random.choice([10, 30, 45])
        gamma = random.choice([0.25,0.5,0.75])
        filter = cv2.getGaborKernel(ksize=(9,9), sigma=sigma, theta=theta, lambd=lambd, gamma=gamma)
        filters.append(np.repeat([filter], channels, axis=0))
    return np.array(filters)


def display_filters(filters):
    '''
    Displays the filters image one by one. Since a 3*3 size is pretty small to
    visualize, filters are rescaled for visualization.
    :param filters: Filters that needs to be visualized
    :return: None
    '''
    subplot_columns = int(len(filters)**0.5)
    fig, axs = plt.subplots(subplot_columns, subplot_columns, figsize=(12,12))
    fig.subplots_adjust(hspace=.01, wspace=.01)
    fig.patch.set_facecolor('black')
    axs = axs.ravel()
    for i in range(len(filters)):
        axs[i].imshow(cv2.resize(filters[i][0],(20,20)), cmap=cm.Greys_r, origin="lower")
        # cv2.imshow("Gabor Kernel", cv2.resize(filters[i][0],(3,3)))
        # cv2.waitKey()
        axs[i].axis('off')
    # cv2.destroyAllWindows()
    plt.show()


receptive_filters = get_gabor_filters(64,1)

display_filters(receptive_filters)

# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train = x_train.astype('float32')/255
# x_test = x_test.astype('float32')/255
#
# y_train = to_categorical(y_train, num_classes=10)
# y_test = to_categorical(y_test, num_classes=10)
#
#
# x_train, _, y_train, _ = train_test_split(x_train,y_train, train_size=0.5, stratify=y_train)
# x_test, _, y_test, _ = train_test_split(x_test, y_test, test_size=0.5, stratify=y_test)
#
# input_dim = (32,32,3)
# input_img = Input(shape=input_dim)
#
# cl1_template = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same',
#              input_shape=input_dim, activation='relu')
# cl1 = cl1_template(input_img)
# # settings the initial filters to gabor filters
# receptive_weights = cl1_template.get_weights()
# # since the order of filter is channel last (3,3, channels, num_filters) so
# # adjusting our gabor filters size according to that format
# receptive_weights[0] = np.moveaxis(receptive_filters, [0,1],[3,2])
# cl1_template.set_weights(receptive_weights)
#
# pl1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(cl1)
# cl2 = Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same',
#              activation='relu')(pl1)
# pl2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(cl2)
#
# cl3 = Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same',
#              activation='relu')(pl2)
# pl3 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(cl3)
#
# cl4 = Conv2D(512, kernel_size=(3,3), strides=(1,1), padding='same',
#              activation='relu')(pl3)
# pl4 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(cl4)
#
# cl5 = Conv2D(512, kernel_size=(3,3), strides=(1,1), padding='same',
#              activation='relu')(pl4)
# pl5 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(cl5)
#
# flat = Flatten()(pl5)
#
# fc6 = Dense(units=4096, activation='relu')(flat)
# fc7 = Dense(units=4096, activation='relu')(fc6)
#
# output = Dense(units=10, activation='softmax')(fc7)
#
# classifier = Model(input_img, output)
#
# opt = RMSprop(learning_rate=0.001)
# classifier.compile(optimizer=opt, loss ='binary_crossentropy',
#                    metrics = ['accuracy'])
# print(classifier.summary())
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
#                               patience=10, min_delta=1e-4,
#                               mode='min', verbose=1)
# stop_alg = EarlyStopping(monitor='val_loss', patience=35,
#                          restore_best_weights=True, verbose=1)
#
# hist = classifier.fit(x_train, y_train, batch_size=100,
#                       epochs=1000, callbacks=[stop_alg, reduce_lr],
#                       shuffle=True, validation_data=(x_test, y_test))
# classifier.save_weights("cnn.hdf5")
#
#
#
