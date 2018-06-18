#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 20:35:54 2018

@author: fuzijie
"""
import os
from datetime import datetime, timedelta
from pandas import read_hdf
from progressbar import ProgressBar
import numpy as np
from sklearn.preprocessing import scale

fd = os.listdir('/Users/fuzijie/Desktop/key/Python-files/cnn/stock_data_with_vwap')

train_period = 60
test_period = 5
num_classes = 3

#dt = '20160701'
#time = '0930'
#dttime = dt + time
#dttime = str(datetime.strptime(dttime, '%Y%m%d%H%M'))
#dt = datetime.strptime(dt,'%Y%m%d')
#dt+timedelta(days=1)
#sample = read_hdf('stocks.h5', 's300271')
#sample
#sample[str(dt.date())]

#dt_initial = '20160630'
#dt_initial = datetime.strptime(dt_initial,'%Y%m%d')

#hdf = HDFStore('returns.h5')
#progress = ProgressBar()
#for stock in progress(fd):
#    stock1 = 's' + stock.replace('.csv','')
#    temp = read_hdf('stocks.h5', stock1, columns = ['stockid', 'time', 'closeprice'])
#    dt = dt_initial
#    rr = pd.DataFrame()
#    for date in range(365):
#        dt = dt + timedelta(days = 1)
#        try:
#            temp1 = temp[str(dt.date())]
#        except:
#            continue
#        else:
#            sp_cprice = np.array(temp1['closeprice'].iloc[0:(240-test_period+1)])
#            temp1 = temp1.iloc[(test_period-1):240]
#            temp1['closeprice'] = temp1['closeprice']/sp_cprice - 1
#            temp1 = temp1.rename(columns = {'closeprice': 'returns'})
#            temp1.index = pd.to_datetime(temp1.time.astype(str))
#            tbname = 'dt' + datetime.strftime(dt, '%Y%m%d') + stock1
#    rr.append(temp1)
#    hdf.put(tbname, temp1, format = 'table', data_columns = temp1.columns)
#hdf.close()
#aaa = read_hdf('returns.h5','dt20160708s603636')

#z = read_hdf('stocks.h5','s000034',columns=['closeprice'])
#z = z.rename(columns = {'closeprice':'returns'})
#z = z/z.shift(4)-1
#z = z.T
#z.index.name = '000034'

dt_initial = '20160630'
dt_initial = datetime.strptime(dt_initial, '%Y%m%d')

x = []
y = []
fd = fd[:5]
progress = ProgressBar()
for stock in progress(fd):
    stock1 = 's' + stock.replace('.csv', '')
    temp = read_hdf('stocks.h5', stock1, columns = ['closeprice', 'volume', 'wb', 'bo', 'vwap'])
    ty = read_hdf('returns.h5', 'returns', columns = [stock1])
    dt = dt_initial
    x0 = np.zeros((1, 103, train_period, 4), dtype = 'float32')
    x0 = (x0.squeeze()).tolist()
    y0 = np.zeros((1, 1), dtype = 'float32')
    for date in range(14):
        dt = dt + timedelta(days = 1)
        try:
            temp1 = temp[str(dt.date())]
        except:
            continue
        else:
            if temp1.empty:
                continue
            tl = temp1.index
            temp1 = scale(temp1)
            for sample in range(len(temp1) - train_period - test_period):
                temp2 = temp1[sample:(sample + train_period),:]
                [smin, smax] = [temp2[:, 0].min(), temp2[:, 0].max()]
                pos = np.rint((temp2[:, 0] - smin)/(smax - smin)*100).astype('int')
                if smax == smin:
                    pos = np.zeros(len(pos), dtype = 'int') + 51
                pos = pos + 2
                ty1 = int(ty.loc[tl[sample+train_period+test_period]])
                for stamp in range(train_period):
                    x0[pos[stamp]-1][stamp] = temp2[stamp, 1:5].tolist()
                    x0[pos[stamp]][stamp] = temp2[stamp, 1:5].tolist()
                    x0[pos[stamp]+1][stamp] = temp2[stamp, 1:5].tolist()
#                x0 = (x0.squeeze()).tolist()
                x.append(x0)
#                x = np.concatenate((x, x0), axis = 0)
                y.append(ty1)

#x = np.delete(x, 0, axis = 0)
x = np.array(x)
y = np.array(y)
y = y.reshape((len(y),1))

