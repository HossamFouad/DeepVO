# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 14:08:00 2018

@author: HOSSAM ABDELHAMID
"""

from scipy.io import loadmat
x = loadmat('perm.mat')
print(x['y'][0][1])