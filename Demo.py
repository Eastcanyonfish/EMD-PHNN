#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 23:53:24 2024

@author: min
"""
import sys
sys.path.append("/Users/min/Documents/EMD-PHNN")


import pandas as pd
A_data=pd.read_excel("/Users/min/Documents/EMD-PHNN/Data/idx.xlsx")
A_data=A_data.loc[3937:7201,["hipr","oppr","clpr","lopr"]]   #  Select A-share data from 2007 to 2021
A_data=A_data.reset_index(drop=True)   
A_data.head()


# use emd to decompose the data into highcomp and lowcomp
from dcomp import  decomp

decompdata = decomp(A_data)
decompdata.process_data()

decompdata.plot_data()
import matplotlib.pyplot as plt
plt.plot(decompdata.highcomp[0])


# Execute DataPreprocessing
from datapreproc import  datapreproc
prcomp=datapreproc(decompdata.lowcomp, decompdata.highcomp, A_data["clpr"])
x_data_trend,x_data_short_term,y = prcomp.preprocess_data()

# Generate the train data and test data
from sklearn.model_selection import train_test_split
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x_data_short_term, y, test_size=0.2, shuffle=True, random_state=0)
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x_data_trend, y, test_size=0.2, shuffle=True, random_state=0)


# train the model
from PHNNmodel import  PHNNmodel
model_type = "CNN-LSTM"  # Choose the desired model type
hybrid_nn = PHNNmodel(model_type)
hybrid_nn.build_model()
hybrid_nn.train_model(x_train_1, x_train_2, y_train_1)