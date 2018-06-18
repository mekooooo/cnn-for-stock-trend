#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 20:33:37 2018

@author: fuzijie
"""

import pandas as pd
import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D, MaxPooling3D
from keras import backend as K





fd = os.listdir('./Sample Data/20160701')
fd.sort()
z = pd.read_csv('./Sample Data/20160701/0931.csv',sep=',', index_col=None)
z = z.drop(z.columns[0], axis=1)
cp_index = list(z.columns)
cp_index = cp_index.index('closeprice')

train_period = 30
test_period = 5
num_classes = 3
batch_size = 5
epochs = 1

str1 = './Sample Data/20160701/'
raw = np.zeros((z.shape[0],z.shape[1],len(fd)))

for num in range(len(fd)):
    str2 = fd[num]
    file_dic = str1 + str2
    z1 = pd.read_csv(file_dic, sep = ',', index_col=None)
    z1 = z1.drop(z1.columns[0], axis=1)
    raw[:,:,num] = z1

x = np.zeros((len(fd)-train_period-test_period,z.shape[0],z.shape[1],train_period))
y = np.zeros((len(fd)-train_period-test_period,z.shape[0]))
for i in range(len(fd)-train_period-test_period):
    x[i,:,:,:] = raw[:,:,i:(train_period+i)]
    temp = raw[:,:,(i+train_period):(i+train_period+test_period)]
    temp = temp[:,cp_index,test_period-1]/temp[:,cp_index,0]-1
    over = np.percentile(temp,66.6)
    under = np.percentile(temp,33.3)
    y[i,:][temp>over] = -1.0
    y[i,:][(temp<=over)&(temp>=under)] = 0.0
    y[i,:][temp<under] = 1.0

## Model ##
if K.image_data_format() == 'channels_first':
    x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2],x.shape[3])
    input_shape = (1,x.shape[1],x.shape[2],x.shape[3])
else:
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2],x.shape[3], 1)
    input_shape = (x.shape[1],x.shape[2],x.shape[3],1)

y_train = keras.utils.to_categorical(y, num_classes)

model = Sequential()
model.add(Conv3D(32, kernel_size=(1, 2, x.shape[3]),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv3D(32, (1,2,1), activation='relu'))
model.add(MaxPooling3D(pool_size=(1, 2, 1)))
model.add(Dropout(0.25))

model.add(Conv3D(64, (1, x.shape[2],x.shape[3]), padding='same', activation='relu'))
model.add(Conv3D(64, (1, 3, 1), activation='relu'))
model.add(MaxPooling3D(pool_size=(1, 1, 1)))
model.add(Dropout(0.25))

model.add(Dense(128, activation='relu',output_shape))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x, y_train))
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

