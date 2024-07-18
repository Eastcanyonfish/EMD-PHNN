#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 08:22:22 2024

@author: min
"""

import numpy as np
from sklearn.model_selection import train_test_split

class datapreproc:
    def __init__(self, lowcomp, highcomp, closepr):
        self.lowcomp = np.array(lowcomp)
        self.highcomp = np.array(highcomp)
        self.closepr = closepr

    def preprocess_data(self):
        # Sample discretely lowcomp to create trend component for x
        lowcomp_t = self.lowcomp.T
        x_data_trend = self._sample_component(lowcomp_t,offset=5)

        # Sample successively highcomp to create short term component for x
        highcomp_t = self.highcomp.T
        x_data_short_term = self._sample_component(highcomp_t, offset=1)

        # Generate labeled variable y for full data
        ret = self.closepr / self.closepr.shift(1) - 1
        y = (ret[16:].to_numpy() >= 0)
        
        return x_data_trend,x_data_short_term,y

      
    def _sample_component(self, data, offset=1):
        (r, c) = data.shape
        x_data = None
        for k in range(r - 16):
            cpat = data[[k - 3*offset+15, k - 2*offset + 15, k - 1*offset + 15, k - 0*offset + 15], :].T
            cpat = (cpat - cpat.min()) / (cpat.max() - cpat.min() + 1)
            if k == 0:
                x_data = cpat
            else:
                x_data = np.append(x_data, cpat, axis=0)
        new_data = x_data.reshape(-1, 4, 4)
        return new_data