#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 00:30:03 2018

@author: fuzijie
"""

import pandas as pd
import matplotlib

a = [1,2,3,4]
b = [15,20,4,9]
p = pd.DataFrame([a,b]).T
p.plot()
