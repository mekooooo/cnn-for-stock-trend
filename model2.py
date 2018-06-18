#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:11:45 2018

@author: fuzijie
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import keras
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, concatenate
from sklearn.cross_validation import train_test_split
#import matplotlib.pyplot as plt

batch_size = 128
num_classes = 3
epochs = 15

# input image dimensions
img_rows = 71
img_cols = 71
train_period = 60
test_period = 5

# the data, shuffled and split between train and test sets

D = np.load('data2.npz')

x_close = D['x_close'].astype('float32')
x_volume = D['x_volume'].astype('float32')
x_cr = D['x_cr'].astype('float32')
x_kdj = D['x_kdj'].astype('float32')
x_boll = D['x_boll'].astype('float32')
y = D['y']

x_close = (255 - x_close) / 255
x_volume = (255 - x_volume) / 255
x_cr = (255 - x_cr) / 255
x_kdj = (255 - x_kdj) / 255
x_boll = (255 - x_boll) / 255

(x_close_train, x_close_test, y_train, y_test) = train_test_split(x_close, y, test_size=0.15, random_state=40)
(x_volume_train, x_volume_test) = train_test_split(x_volume, test_size=0.15, random_state=40)
(x_cr_train, x_cr_test) = train_test_split(x_cr, test_size=0.15, random_state=40)
(x_kdj_train, x_kdj_test) = train_test_split(x_kdj, test_size=0.15, random_state=40)
(x_boll_train, x_boll_test) = train_test_split(x_boll, test_size=0.15, random_state=40)


input_shape = x_close.shape[1:]
print('Images Input Shape:', input_shape)
#print(x_close_train.shape[0], 'train samples')
#print(x_close_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#yy = keras.utils.to_categorical(y, num_classes)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# CNN model for closeprice images
input_close = Input(shape = input_shape, dtype = 'float32', name = 'close_input')
c1_close = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(input_close)
#c1_close = Conv2D(32, (3, 3), activation = 'relu')(c1_close)
m1_close = MaxPooling2D(pool_size = (2, 2))(c1_close)
dp1_close = Dropout(0.25)(m1_close)

c2_close = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(dp1_close)
#c2_close = Conv2D(64, (3, 3), activation = 'relu')(c2_close)
m2_close = MaxPooling2D(pool_size=(2, 2))(c2_close)
dp2_close = Dropout(0.25)(m2_close)

#c2_close = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')(dp2_close)
##c2_close = Conv2D(64, (3, 3), activation = 'relu')(c2_close)
#m2_close = MaxPooling2D(pool_size=(2, 2))(c2_close)
#dp2_close = Dropout(0.25)(m2_close)

f_close = Flatten()(dp2_close)
d1_close = Dense(512, activation='relu')(f_close)

# CNN model for volume images
input_volume = Input(shape = input_shape, dtype = 'float32', name = 'volume_input')
c1_volume = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(input_volume)
#c1_volume = Conv2D(32, (3, 3), activation = 'relu')(c1_volume)
m1_volume = MaxPooling2D(pool_size = (2, 2))(c1_volume)
dp1_volume = Dropout(0.25)(m1_volume)

c2_volume = Conv2D(64, (3, 3), padding = 'same', activation='relu')(dp1_volume)
#c2_volume = Conv2D(64, (3, 3), activation = 'relu')(c2_volume)
m2_volume = MaxPooling2D(pool_size=(2, 2))(c2_volume)
dp2_volume = Dropout(0.25)(m2_volume)

#c2_volume = Conv2D(64, (3, 3), padding = 'same', activation='relu')(dp2_volume)
##c2_volume = Conv2D(64, (3, 3), activation = 'relu')(c2_volume)
#m2_volume = MaxPooling2D(pool_size=(2, 2))(c2_volume)
#dp2_volume = Dropout(0.25)(m2_volume)

f_volume = Flatten()(dp2_volume)
d1_volume = Dense(100, activation='relu')(f_volume)

# CNN model for CR images
input_cr = Input(shape = input_shape, dtype = 'float32', name = 'cr_input')
c1_cr = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(input_cr)
#c1_cr = Conv2D(32, (3, 3), activation = 'relu')(c1_cr)
m1_cr = MaxPooling2D(pool_size = (2, 2))(c1_cr)
dp1_cr = Dropout(0.25)(m1_cr)

c2_cr = Conv2D(64, (3, 3), padding = 'same', activation='relu')(dp1_cr)
#c2_cr = Conv2D(64, (3, 3), activation = 'relu')(c2_cr)
m2_cr = MaxPooling2D(pool_size=(2, 2))(c2_cr)
dp2_cr = Dropout(0.25)(m2_cr)

#c2_cr = Conv2D(64, (3, 3), padding = 'same', activation='relu')(dp2_cr)
##c2_cr = Conv2D(64, (3, 3), activation = 'relu')(c2_cr)
#m2_cr = MaxPooling2D(pool_size=(2, 2))(c2_cr)
#dp2_cr = Dropout(0.25)(m2_cr)

f_cr = Flatten()(dp2_cr)
d1_cr = Dense(512, activation='relu')(f_cr)

# CNN model for KDJ images
input_kdj = Input(shape = input_shape, dtype = 'float32', name = 'kdj_input')
c1_kdj = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(input_kdj)
#c1_kdj = Conv2D(32, (3, 3), activation = 'relu')(c1_kdj)
m1_kdj = MaxPooling2D(pool_size = (2, 2))(c1_kdj)
dp1_kdj = Dropout(0.25)(m1_kdj)

c2_kdj = Conv2D(64, (3, 3), padding = 'same', activation='relu')(dp1_kdj)
#c2_kdj = Conv2D(64, (3, 3), activation = 'relu')(c2_kdj)
m2_kdj = MaxPooling2D(pool_size=(2, 2))(c2_kdj)
dp2_kdj = Dropout(0.25)(m2_kdj)

#c2_kdj = Conv2D(64, (3, 3), padding = 'same', activation='relu')(dp2_kdj)
##c2_kdj = Conv2D(64, (3, 3), activation = 'relu')(c2_kdj)
#m2_kdj = MaxPooling2D(pool_size=(2, 2))(c2_kdj)
#dp2_kdj = Dropout(0.25)(m2_kdj)

f_kdj = Flatten()(dp2_kdj)
d1_kdj = Dense(512, activation='relu')(f_kdj)

# CNN model for BOLL images
input_boll = Input(shape = input_shape, dtype = 'float32', name = 'boll_input')
c1_boll = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(input_boll)
#c1_boll = Conv2D(32, (3, 3), activation = 'relu')(c1_boll)
m1_boll = MaxPooling2D(pool_size = (2, 2))(c1_boll)
dp1_boll = Dropout(0.25)(m1_boll)

c2_boll = Conv2D(64, (3, 3), padding = 'same', activation='relu')(dp1_boll)
#c2_boll = Conv2D(64, (3, 3), activation = 'relu')(c2_boll)
m2_boll = MaxPooling2D(pool_size=(2, 2))(c2_boll)
dp2_boll = Dropout(0.25)(m2_boll)

#c2_boll = Conv2D(64, (3, 3), padding = 'same', activation='relu')(dp2_boll)
##c2_boll = Conv2D(64, (3, 3), activation = 'relu')(c2_boll)
#m2_boll = MaxPooling2D(pool_size=(2, 2))(c2_boll)
#dp2_boll = Dropout(0.25)(m2_boll)

f_boll = Flatten()(dp2_boll)
d1_boll = Dense(512, activation='relu')(f_boll)

# take Dense output into LSTM
c_input = concatenate([d1_close, d1_volume, d1_cr, d1_kdj, d1_boll])
d2 = Dense(128, activation = 'relu')(c_input)
d2 = Dropout(0.25)(d2)
#d2 = Dense(64, activation = 'relu')(d2)
output = Dense(num_classes, activation = 'softmax')(d2)

model = Model(inputs = [input_close, input_volume, input_cr, input_kdj, input_boll],
              outputs = output)
print(model.summary())

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit({'close_input': x_close_train, 'volume_input': x_volume_train,
                     'cr_input': x_cr_train, 'kdj_input': x_kdj_train,
                     'boll_input': x_boll_train}, y_train, batch_size=batch_size,
                      validation_data=({'close_input': x_close_test,
                     'volume_input': x_volume_test, 'cr_input': x_cr_test, 
                     'kdj_input': x_kdj_test, 'boll_input': x_boll_test}, 
                      y_test),epochs=epochs, verbose=1)
#score = model.evaluate({'close_input': x_close_test, 'volume_input': x_volume_test,
#                     'cr_input': x_cr_test, 'kdj_input': x_kdj_test,
#                     'boll_input': x_boll_test}, y_test)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
#
## Loss Curves
#plt.figure(figsize=[7,6])
#plt.plot(history.history['loss'],'r',linewidth=3.0)
#plt.plot(history.history['val_loss'],'b',linewidth=3.0)
#plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
#plt.xlabel('Epochs ',fontsize=16)
#plt.ylabel('Loss',fontsize=16)
#plt.title('Loss Curves',fontsize=16)
## 
### Accuracy Curves
#plt.figure(figsize=[7,6])
#plt.plot(history.history['acc'],'r',linewidth=3.0)
#plt.plot(history.history['val_acc'],'b',linewidth=3.0)
#plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
#plt.xlabel('Epochs ',fontsize=16)
#plt.ylabel('Accuracy',fontsize=16)
#plt.title('Accuracy Curves',fontsize=16)
