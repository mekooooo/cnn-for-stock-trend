#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 22:08:33 2018

@author: fuzijie
"""

import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from pandas import read_hdf
import os
import io
import stockstats
from progressbar import ProgressBar
import zipfile
import time
import keras
import glob
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, concatenate
from sklearn.cross_validation import train_test_split
#import matplotlib.pyplot as plt

batch_size = 128
num_classes = 3
epochs = 15

# input image dimensions
img_rows = 80
img_cols = 80
train_period = 60
test_period = 5
input_shape = (img_rows, img_cols, num_classes)


def generate_data_from_file(path, folder, start, end):
    bt_close = []
    bt_volume = []
    bt_cr = []
    bt_kdj = []
    bt_boll = []
    archive = zipfile.ZipFile(path, 'r')
    y = np.load(io.BytesIO(archive.read(folder + '/returns_spy.npz')))['y'].tolist()
    y = np.array(y)
    y = y.reshape((len(y), 1))
    y = keras.utils.to_categorical(y, 3)
    y = y[start:end]
    progress = ProgressBar()
    for i in progress(range(start,end)):
        xt = plt.imread(io.BytesIO(archive.read(folder + '/close/' + str(i+1) + '.png')))
        bt_close.append(np.delete(xt,3,2).tolist());del xt
#        xt1 = np.delete(xt1,3,2)
        xt = plt.imread(io.BytesIO(archive.read(folder + '/volume/' + str(i+1) + '.png')))
        bt_volume.append(np.delete(xt,3,2).tolist());del xt
#        xt2 = np.delete(xt2,3,2)
        xt = plt.imread(io.BytesIO(archive.read(folder + '/cr/' + str(i+1) + '.png')))
        bt_cr.append(np.delete(xt,3,2).tolist());del xt
#        xt3 = np.delete(xt3,3,2)
        xt = plt.imread(io.BytesIO(archive.read(folder + '/kdj/' + str(i+1) + '.png')))
        bt_kdj.append(np.delete(xt,3,2).tolist());del xt
#        xt4 = np.delete(xt4,3,2)
        xt = plt.imread(io.BytesIO(archive.read(folder + '/boll/' + str(i+1) + '.png')))
        bt_boll.append(np.delete(xt,3,2).tolist());del xt
#        xt5 = np.delete(xt5,3,2)
#        yt = y[i].astype('float32')
#        yield ({'close_input': xt1.astype('float32').reshape((1,xt1.shape[0],xt1.shape[1],xt1.shape[2])),
#               'volume_input': xt2.astype('float32').reshape((1,xt2.shape[0],xt2.shape[1],xt2.shape[2])),
#               'cr_input': xt3.astype('float32').reshape((1,xt3.shape[0],xt3.shape[1],xt3.shape[2])),
#               'kdj_input': xt4.astype('float32').reshape((1,xt4.shape[0],xt4.shape[1],xt4.shape[2])),
#               'boll_input': xt5.astype('float32').reshape((1,xt5.shape[0],xt5.shape[1],xt5.shape[2]))}, 
#               yt.reshape(1,3))
    bt_close = np.array(bt_close, dtype = 'float32')
    bt_volume = np.array(bt_volume, dtype = 'float32')
    bt_cr = np.array(bt_cr, dtype = 'float32')
    bt_kdj = np.array(bt_kdj, dtype = 'float32')
    bt_boll = np.array(bt_boll, dtype = 'float32')
    return [bt_close, bt_volume, bt_cr, bt_kdj, bt_boll, y]


[x_close,x_volume,x_cr,x_kdj,x_boll,y] = generate_data_from_file('./plots.zip', 'plots5', 0, 3150)
np.savez('plots5.npz', x_close = x_close, x_volume = x_volume, 
         x_cr = x_cr, x_kdj = x_kdj, x_boll = x_boll, y = y)

glob.glob('./plots/plots1/boll/*.png')


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


model.train_on_batch({'close_input': x_close, 'volume_input': x_volume,
                     'cr_input': x_cr, 'kdj_input': x_kdj,
                     'boll_input': x_boll},y)
g = model.fit_generator(generate_data_from_file('./plots.zip', 'plots1', 0, 3150), steps_per_epoch = 315, epochs =10)

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
