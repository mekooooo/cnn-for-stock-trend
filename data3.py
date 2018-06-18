#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 18:40:12 2018

@author: fuzijie
"""

import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from pandas import read_hdf
import os
import stockstats
from progressbar import ProgressBar

matplotlib.use('Agg')

def make_image(data, outputname, size=(1, 1), dpi=80):
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.plot(data)
    plt.savefig(outputname, dpi=dpi)
    plt.close()

fd = os.listdir('/Users/fuzijie/Desktop/key/Python-files/cnn/stock_data_with_vwap')
train_period = 60
test_period = 5
allreturns = read_hdf('./returns.h5', 'returns')
cflag = 1
rid = np.random.permutation(len(fd))
y = []
fd = [fd[i] for i in rid[:3]]
for stock in fd:
    data = pd.read_csv('./stock_data_with_vwap/'+ stock, dtype = {'stockid': str})
    data.index = pd.to_datetime(data.time.astype(str))
    data = data.rename(columns = {'closeprice':'close','highprice':'high','lowprice':'low','openprice':'open'})
    ty = allreturns['s' + stock.replace('.csv', '')]
    data['date'] = data.index.date
    progress = ProgressBar()
    for cut in progress([32,34,35,37,39,41]):
        temp = data[(cut*240):((cut+1)*240)]
        if temp.empty or len(np.unique(temp['close'])) < 10:
            continue
        tty = ty[ty.index.date == temp.index.date[1]]
        temp = stockstats.StockDataFrame.retype(temp)
        temp.index = range(len(temp))
        _ = temp[['cr','kdjk','boll']]
        if sum(np.sum(temp == float('inf')))>0:
            temp = temp.replace([float('inf'),float('-inf')],0)
            temp['cr'] = temp['cr'].replace(0, temp['cr'].mean())
        for sample in range(len(temp)-train_period-test_period):
            imname = str(cflag) + '.png'
            make_image(temp[sample:(sample + train_period)]['close'], './plots5/close/' + imname, dpi = 80)
            make_image(temp[sample:(sample + train_period)]['volume'], './plots5/volume/' + imname, dpi = 80)
            make_image(temp[sample:(sample + train_period)][['cr', 'cr-ma1', 'cr-ma2', 'cr-ma3']], './plots5/cr/' + imname, dpi = 80)
            make_image(temp[sample:(sample + train_period)][['kdjk', 'kdjd', 'kdjj']], './plots5/kdj/' + imname, dpi = 80)
            make_image(temp[sample:(sample + train_period)][['boll_ub', 'boll_lb']], './plots5/boll/' + imname, dpi = 80)
            y.append(tty[len(tty) - 240 + sample + train_period + test_period])
            cflag += 1

y = np.array(y) 
y = y.reshape((len(y),1))
ytt=[]
np.savez('./plots5/returns_spy.npz', y = y)
yt = np.load('./plots5/returns_spy.npz')
yt = yt['y'].tolist()
ytt.extend(yt)
ccc = []
for i in range(1000):
    ccc.append(plt.imread('./plots5/close/1.png').tolist())

ccc = np.array(ccc)
ccc.append(ccc)

import zipfile
import io
archive = zipfile.ZipFile('plots.zip', 'r')
imgdata = plt.imread(io.BytesIO(archive.read('plots1/close/1.png')))
ii = np.delete(imgdata,3,2)
archive.namelist()[7]
