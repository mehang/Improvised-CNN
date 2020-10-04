import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# dimensionality of input and latent encoded representations
input_dim = (32,32,3)

input_img = Input(shape=input_dim)

# Convolutional layer
cl1 = Conv2D(64, (9,9), strides=(2,2), input_shape=input_dim,
             activation='relu')(input_img)

# Pooling and Batch normalization
pl2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(cl1)
bnl3 = BatchNormalization()(pl2)

# Add second layer of convolutional layer
cl4 = Conv2D(123, (3,3), strides=(1,1),
             activation='relu')(bnl3)
pl5 = MaxPooling2D(pool_size=(2,2))(cl4)
bnl6 = BatchNormalization()(pl5)

# FLattening for compatibility
fl7 = Flatten()(bnl6)

# Denser layers + Dropout
dol8 = Dropout(0.5)(fl7)
dl9 = Dense(units=256, activation='relu')(dol8)
dol10 = Dropout(0.2)(dl9)
dl11 = Dense(units=64, activation='relu')(dol10)
dol12 = Dropout(0.)(dl11)
output = Dense(units=10, activation='sigmoid')(dol12)

classifier = Model(input_img, output)

# Compiling the CNN with RMSprop optimizer
opt = RMSprop(learning_rate=0.001)

classifier.compile(optimizer=opt, loss ='binary_crossentropy',
                   metrics = ['accuracy'])
print(classifier.summary())

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=10, min_delta=1e-4,
                              mode='min', verbose=1)
stop_alg = EarlyStopping(monitor='val_loss', patience=35,
                         restore_best_weights=True, verbose=1)

hist = classifier.fit(x_train, y_train, batch_size=100,
                      epochs=1000, callbacks=[stop_alg, reduce_lr],
                      shuffle=True, validation_data=(x_test, y_test))
classifier.save_weights("cnn.hdf5")



