#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:22:02 2024

@author: min
"""

import numpy as np
from PyEMD import EMD
from scipy import stats
import matplotlib.pyplot as plt

class decomp:
    def __init__(self, data):
        self.data = data
        self.highcomp = []
        self.lowcomp = []

    def decompose(self, series):
        S = series
        emd = EMD()
        emd.emd(S)
        imfs, res = emd.get_imfs_and_residue()
        return imfs, res

    def reconstitution(self, imfs, res, S):
        sample = np.zeros(imfs[0].shape)
        for i in range(len(imfs)):
            sample = sample + imfs[i]
            sample = np.asarray(sample)
            r = stats.ttest_1samp(sample, 0, axis=0)
            if r.pvalue < 0.05:
                series1 = sample - imfs[i]
                series2 = S - series1
                break
        return series1, series2

    def process_data(self):
        for i in range(4):
            series = self.data[self.data.columns[i]].values
            imfs, res = self.decompose(series)
            series1_, series2_ = self.reconstitution(imfs, res, series)
            self.highcomp.append(series1_)
            self.lowcomp.append(series2_)

    def plot_data(self):
        plt.plot(self.highcomp[0])
        plt.plot(self.lowcomp[0])

