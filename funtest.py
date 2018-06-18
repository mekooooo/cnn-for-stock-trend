#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 18:16:46 2018

@author: fuzijie
"""

import matplotlib.pyplot as plt
import pylab
import imageio
import skimage.io
import numpy as np
from PIL import Image
import time
import io
from io import StringIO
import PIL
from matplotlib.figure import Figure

#申请缓冲地址
buffer_ = StringIO()#using buffer,great way!
#保存在内存中，而不是在本地磁盘，注意这个默认认为你要保存的就是plt中的内容

plt.savefig(buffer_,format = 'png')
buffer_.seek(0)
#用PIL或CV2从内存中读取
dataPIL = PIL.Image.open(buffer_)
#转换为nparrary，PIL转换就非常快了,data即为所需
data = np.asarray(dataPIL)
#释放缓存    
buffer_.close()



plt.figure()
plt.plot([1, 2])
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
im = Image.open(buf)
im.show()
buf.close()

fig = Figure(figsize=(1, 1))
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(sdata['close'])

buf = io.BytesIO()
fig = Figure(figsize=(1,1))
ax = fig.add_axes([0, 0, 1, 1])
ax.plot(temp['close'])
ax.savefig(buf, format='png')
buf.seek(0)
im = Image.open(buf)
im = np.array(im.getdata()).reshape(im.size[1], im.size[0], 4)
plt.close()
buf.close()

def Closeprice_plot_array1(sdata):
    buf = StringIO()
    plt.figure()
    plt.plot(sdata['close'])
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf)
    plt.close()
    buf.close()
    return np.array(im)

def Construct_plot_array1(sdata, fun, train_period = 60, test_period = 5):
    result = list(range(len(sdata)-train_period-test_period))
    for sample in range(len(sdata)-train_period-test_period):
        result[sample] = fun(sdata[sample:(sample + train_period)])
    return result

tic = time.time()
result = list(range(175))
for i in range(175):
    result[i]= Closeprice_plot_array(temp[:60])
toc = time.time()
print('Time cost:', toc-tic, 'seconds')


tic = time.time()
for i in range(5):
    xx = Construct_plot_array1(temp,fun = Closeprice_plot_array)
#    xx = Closeprice_plot_array(temp)
    xx = np.array(xx)
toc = time.time()
print('Shape of one image:', xx[2].shape)
print('Time cost:', toc-tic, 'seconds')

tic = time.time()
for i in range(5):
    xx = Construct_plot_array(temp,fun = Closeprice_plot_array)
#    xx = Closeprice_plot_array(temp)
    xx = np.array(xx)
toc = time.time()
print('Shape of one image:', xx[2].shape)
print('Time cost:', toc-tic, 'seconds')