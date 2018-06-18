#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 18:14:12 2018

@author: fuzijie
"""

x_close1 = []
x_close2 = []
x_close3 = []
x_close4 = []
x_close5 = []

progress = ProgressBar()
for i in progress(range(3150)):
  xt = plt.imread(io.BytesIO(archive.read('plots1/close/'+str(i+1)+'.png')))
  x_close1.append(np.delete(xt,3,2).tolist());del xt
progress = ProgressBar()
for i in progress(range(2625)):
  xt = plt.imread(io.BytesIO(archive.read('plots2/close/'+str(i+1)+'.png')))
  x_close2.append(np.delete(xt,3,2).tolist());del xt
progress = ProgressBar()
for i in progress(range(2975)):
  xt = plt.imread(io.BytesIO(archive.read('plots3/close/'+str(i+1)+'.png')))
  x_close3.append(np.delete(xt,3,2).tolist());del xt
progress = ProgressBar()
for i in progress(range(3150)):
  xt = plt.imread(io.BytesIO(archive.read('plots4/close/'+str(i+1)+'.png')))
  x_close4.append(np.delete(xt,3,2).tolist());del xt
progress = ProgressBar()
for i in progress(range(3150)):
  xt = plt.imread(io.BytesIO(archive.read('plots5/close/'+str(i+1)+'.png')))
  x_close5.append(np.delete(xt,3,2).tolist());del xt