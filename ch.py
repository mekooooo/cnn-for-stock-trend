#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 20:49:15 2018

@author: fuzijie
"""

import requests, json
from urllib.parse import unquote
from bs4 import BeautifulSoup as bs


if __name__ == '__main__':
    r = requests.get('http://data.eastmoney.com/rzrq/detail/300059.html')
    t = r.text
    s = bs(r.text, 'lxml')


    url = 'http://dcfm.eastmoney.com/em_mutisvcexpandinterface/api/js/get'
    par = { 'type': 'RZRQ_DETAIL_NJ',
            'token': '70f12f2f4f091e459a279469fe49eca5',
            'filter': unquote('(scode=%27300059%27)'),
            'st': 'tdate',
            'sr': '-1',
            'p': '1',
            'ps': '5000',
            'js': 'var%20JBZRDqWc={pages:(tp),data:(x)}',
            'time': '1',
            'rt': '50778927'
            }
    j = requests.get(url, params = par)
    j = ''.join(j.text.split('=')[1:])
    j = j.replace('data', '"data"')
    j = j.replace('pages', '"pages"')
    js = json.loads(j)




