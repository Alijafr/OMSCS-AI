#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 11:14:16 2021

@author: labuser
"""

import numpy as np 
x_series = [39,52,66,37,69,55,70,20]
#y_series = [43,62,71,40,22,58,64,53,78,25]
y_series = [25,78,53,64,58,22,40,71,62,43]

x_series = np.array(x_series)
y_series = np.array(y_series)

DWT_matrix = np.zeros((y_series.shape[0],x_series.shape[0]))

for i in range(len(DWT_matrix)):
    for j in range(len(DWT_matrix[0])):
        DWT_matrix[i][j]= (y_series[i] - x_series[j])**2
        

squared_root_sum = (DWT_matrix.sum())**0.5

