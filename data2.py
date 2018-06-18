#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:02:07 2018

@author: fuzijie
"""

import pandas as pd
from pandas import read_hdf
import numpy as np
import os
import stockstats
from progressbar import ProgressBar
#from datetime import datetime
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

def Closeprice_plot_array(sdata):
    fig = Figure(figsize=(1, 1))
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(sdata['close'])
    canvas.draw()
    xx = np.frombuffer(canvas.tostring_rgb(),dtype='uint8')
    xx = xx.reshape(fig.canvas.get_width_height()[::-1]+(3,))
    xx = np.delete(xx,0,0)
    xx = np.delete(xx,0,1)
    xx = xx.squeeze().tolist()
    return xx

def Volume_plot_array(sdata):
    fig = Figure(figsize=(1, 1))
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(sdata['volume'])
    canvas.draw()
    xx = np.frombuffer(canvas.tostring_rgb(),dtype='uint8')
    xx = xx.reshape(fig.canvas.get_width_height()[::-1]+(3,))
    xx = np.delete(xx,0,0)
    xx = np.delete(xx,0,1)
    xx = xx.squeeze().tolist()
    return xx

def CR_plot_array(sdata):
    fig = Figure(figsize=(1, 1))
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(sdata[['cr', 'cr-ma1', 'cr-ma2', 'cr-ma3']])
    canvas.draw()
    xx = np.frombuffer(canvas.tostring_rgb(),dtype='uint8')
    xx = xx.reshape(fig.canvas.get_width_height()[::-1]+(3,))
    xx = np.delete(xx,0,0)
    xx = np.delete(xx,0,1)
    xx = xx.squeeze().tolist()
    return xx

def KDJ_plot_array(sdata):
    fig = Figure(figsize=(1, 1))
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(sdata[['kdjk', 'kdjd', 'kdjj']])
    canvas.draw()
    xx = np.frombuffer(canvas.tostring_rgb(),dtype='uint8')
    xx = xx.reshape(fig.canvas.get_width_height()[::-1]+(3,))
    xx = np.delete(xx,0,0)
    xx = np.delete(xx,0,1)
    xx = xx.squeeze().tolist()
    return xx

def BOLL_plot_array(sdata):
    fig = Figure(figsize=(1, 1))
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(sdata[['boll_ub', 'boll_lb']])
    canvas.draw()
    xx = np.frombuffer(canvas.tostring_rgb(),dtype='uint8')
    xx = xx.reshape(fig.canvas.get_width_height()[::-1]+(3,))
    xx = np.delete(xx,0,0)
    xx = np.delete(xx,0,1)
    xx = xx.squeeze().tolist()
    return xx

def Construct_plot_array(sdata, fun, train_period = 60, test_period = 5):
    result = []
    for sample in range(len(sdata)-train_period-test_period):
        result.append(fun(sdata[sample:(sample + train_period)]))
    return result

def Construct_response(rdata, train_period = 60, test_period = 5):
    return rdata[(len(rdata)-240+train_period+test_period):len(rdata)]


fd = os.listdir('/Users/fuzijie/Desktop/key/Python-files/cnn/stock_data_with_vwap')

train_period = 60
test_period = 5

#dt_initial = '20160630'
#dt_initial = datetime.strptime(dt_initial, '%Y%m%d')


x_close = []
x_volume = []
x_cr = []
x_kdj = []
x_boll = []
y = []
allreturns = read_hdf('./returns.h5', 'returns')
rid = np.random.permutation(len(fd))
#fd = fd[:5]
fd = [fd[i] for i in rid[:3]]
for stock in fd:
    stock1 = 's' + stock.replace('.csv', '')
    data = read_hdf('stocks.h5', stock1, columns = ['closeprice', 'highprice', 'lowprice', 'openprice', 'volume'])
    data = data.rename(columns = {'closeprice':'close','highprice':'high','lowprice':'low','openprice':'open'})
    progress = ProgressBar()
    for cut in progress([5,13,25,17,8]):
        temp = data[(cut*240):((cut+1)*240)]
        if temp.empty:
            continue
        date_flag = temp.index.date
        temp['date_flag'] = date_flag
        temp['date'] = date_flag
        temp = stockstats.StockDataFrame.retype(temp)
        temp.index = range(len(temp))
        _ = temp[['cr','kdjk','boll']]
        if sum(np.sum(temp == float('inf')))>0:
            continue
        x_close += Construct_plot_array(temp,fun = Closeprice_plot_array)
    #    xx = np.vstack(xx).squeeze().tolist()x 
    #    x_close.append(xx)
#        x_close = x_close + xx
        x_volume += Construct_plot_array(temp,fun = Volume_plot_array)
#        x_volume = x_volume + xx
        x_cr += Construct_plot_array(temp,fun = CR_plot_array)
#        x_cr = x_cr + xx
        x_kdj += Construct_plot_array(temp,fun = KDJ_plot_array)
#        x_kdj = x_kdj + xx
        x_boll += Construct_plot_array(temp,fun = BOLL_plot_array)
#        x_boll = x_boll + xx
    #    temp = stockstats.StockDataFrame.retype(temp)
    #    _ = temp[['cr','boll','kdjk']]
    #    temp = stockstats.StockDataFrame.retype(temp)
        ty = allreturns[stock1]
        ty = ty[ty!=-2]
        ty = pd.DataFrame(ty)
        ty['date_flag'] = ty.index.date
        ty = ty[ty['date_flag'].isin(np.unique(date_flag))]
        tty = ty.groupby('date_flag').apply(Construct_response).iloc[:,0].astype('int').tolist()
        y = y + tty

#x = np.delete(x, 0, axis = 0)
x_close = np.array(x_close)
#x_close = np.vstack(x_close)
x_volume = np.array(x_volume)
#x_volume = np.vstack(x_volume)
x_cr = np.array(x_cr)
#x_cr = np.vstack(x_cr)
x_kdj = np.array(x_kdj)
#x_kdj = np.vstack(x_kdj)
x_boll = np.array(x_boll)
#x_boll = np.vstack(x_boll)
y = np.array(y) 
y = y.reshape((len(y),1))

np.savez('data2.npz', x_close = x_close, x_volume = x_volume, 
         x_cr = x_cr, x_kdj = x_kdj, x_boll = x_boll, y = y)
D = np.load('data2.npz')
