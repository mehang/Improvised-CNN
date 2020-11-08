from skimage.filters import gabor_kernel
from skimage import io
from skimage.transform import resize
from matplotlib import pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

import tensorflow as tf

import math

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 100
IMAGE_SIZE = 32
NUM_OF_CLASSES = 10

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train = x_train.astype('float32') / 255.0
# x_test = x_test.astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


def process_dataset(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    #     image = tf.image.per_image_standardization(image)

    #     image = tf.image.convert_image_dtype(image, tf.float32)
    # Resize images from 32x32 to 227x227
    #     image = tf.image.resize(image, (IMAGE_SIZE,IMAGE_SIZE))

    #     image = image/255.0
        print("--------")
        print(image.shape())
        jpt =  tf.image.convert_image_dtype(image, tf.float32)
        print(jpt/255.0)
        print(label)
        # label = tf.one_hot(tf.cast(label, tf.int32), NUM_OF_CLASSES)

        return image, label


train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))


def construct_dataset(ds):
    ds = ds.shuffle(buffer_size=BATCH_SIZE)

    #     ds=ds.repeat()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.map(process_dataset,num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


train_data = construct_dataset(train_data)
test_data = construct_dataset(test_data)

steps_per_epoch = math.ceil(50000 / BATCH_SIZE)
validation_steps = math.ceil(10000 / BATCH_SIZE)
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 100
IMAGE_SIZE = 32
NUM_OF_CLASSES = 10

# Importing the Keras libraries and packages
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Activation
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Adam

# dimensionality of input and latent encoded representations
inpt_dim = (32, 32, 3)

inpt_img = Input(shape=inpt_dim)

rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape = inpt_dim )(inpt_img)

# Block 1
cl1 = Conv2D(64, (3, 3), strides=(2, 2),activation='relu')(rescale)
bnl2 = BatchNormalization()(cl1)
# afl3 = Activation('relu')(bnl2)
pl4 = MaxPooling2D(pool_size = (2, 2))(bnl2)

# Adding a second convolutional layer
cl5 = Conv2D(128, (3, 3), strides=(1, 1), activation='relu')(pl4)
bnl6 = BatchNormalization()(cl5)
# afl7 = Activation('relu')(bnl6)
pl8 = MaxPooling2D(pool_size = (2, 2))(bnl6)
bnl9 = BatchNormalization()(pl8)

# Step 3 - Flattening
fl10 = Flatten()(bnl9)

# Step 4 - Full connection
dol11 = Dropout(0.5)(fl10)
dl12 = Dense(units = 256, activation = 'relu')(dol11)
dol13 = Dropout(0.2)(dl12)
dl14 = Dense(units = 64, activation = 'relu')(dol13)
dol15 = Dropout(0.1)(dl14)
output = Dense(units = 10, activation = 'sigmoid')(dol15)

classifier = Model(inpt_img, output)

# Compiling the CNN
opt = RMSprop(learning_rate=0.001)
# opt = Adam(learning_rate=0.01)

classifier.compile(optimizer = opt, loss ='binary_crossentropy',
                   metrics = ['accuracy'])

print(classifier.summary())

# Fitting the CNN to the images

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
                              min_delta=1e-4, mode='min', verbose=1)

stop_alg = EarlyStopping(monitor='val_loss', patience=35,
                         restore_best_weights=True, verbose=1)



hist = classifier.fit(train_data,  epochs=1000,
                   callbacks=[stop_alg, reduce_lr],
                      validation_steps=validation_steps,
                      steps_per_epoch=steps_per_epoch,
                   validation_data=test_data)

