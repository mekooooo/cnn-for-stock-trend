#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 14:48:47 2018

@author: fuzijie
"""
import os
import pandas as pd
from pandas import HDFStore, read_hdf
from progressbar import ProgressBar

fd = os.listdir('/Users/fuzijie/Desktop/key/Python-files/cnn/stock_data_with_vwap')

hdf = HDFStore('stocks.h5')
progress = ProgressBar()
for stock in progress(fd):
    z = pd.read_csv('./stock_data_with_vwap/' + stock, sep = ',', dtype = {'stockid': str})
    z.index = pd.to_datetime(z.time.astype(str))
    stock = 's' + stock.replace('.csv', '')
#    z.to_hdf('stock.h5',key = stock,mode='a',data_column=['closeprice','openprice'],format='table')
    hdf.put(stock, z, format = 'table', data_columns = z.columns)
hdf.close()

rr = pd.DataFrame()
progress = ProgressBar()
for stock in progress(fd):
    stock1 = 's' + stock.replace('.csv','')
    temp = read_hdf('stocks.h5', stock1, columns = ['closeprice'])
    temp = temp.rename(columns = {'closeprice': stock.replace('.csv','')})
    temp = temp/temp.shift(4) - 1
    temp = temp.T
    rr = rr.append(temp)

rr = rr.drop(rr.columns[:4],axis = 1)
rr = rr.apply(pd.qcut, axis = 0, args = (3, [-1,0,1]))
rr = rr.sort_index(axis = 'index')
rr = rr.T
rr = rr.fillna(-2)
rr.columns = 's' + rr.columns
hdf1 = HDFStore('returns.h5')
hdf1.put('returns', rr, format = 'table', data_columns = rr.columns)
hdf1.close()
