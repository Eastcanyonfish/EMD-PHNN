#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 08:56:04 2024

@author: min
"""



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import *
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

class PHNNmodel:
    def __init__(self, model_type):
        self.model_type = model_type
        self.model = None

    def build_model(self):
        if self.model_type == "CNN-LSTM":
            modelx = Sequential()
            # Define CNN layers
            modelx=Sequential()
            modelx.add(Conv2D(64,(2,2),padding='same',activation='relu',input_shape=(4,4,1)))
            modelx.add(MaxPooling2D(2,2)) 
            modelx.add(Dropout(0.3))
            modelx.add(Flatten())
            modelx.add(Dense(16,activation='elu'))
            modelx.add(Dropout(0.3))
            modelx.add(Flatten())
            #
            modely=Sequential()
            modely.add(LSTM(64, input_shape=(4,4), return_sequences=True))
            modely.add(Dropout(0.3))
            modely.add(LSTM(64, input_shape=(4,4), return_sequences=False))
            modely.add(Dropout(0.3))
            modely.add(Flatten())
            modely.add(Dense(16,activation='elu'))
            modely.add(Dropout(0.3))
            modely.add(Flatten())
            #
            dummy_input_x = Input(shape=(4,4,1))
            dummy_input_y = Input(shape=(4,4))

        elif self.model_type == "LSTM-LSTM":
            # Define LSTM-LSTM model
            modelx=Sequential()
            modelx.add(LSTM(64, input_shape=(4,4), return_sequences=True))
            modelx.add(Dropout(0.3))
            modelx.add(LSTM(64, input_shape=(4,4), return_sequences=False))
            modelx.add(Dropout(0.3))
            modelx.add(Flatten())
            modelx.add(Dense(16,activation='elu'))
            modelx.add(Dropout(0.3))
            modelx.add(Flatten())
            #
            modely=Sequential()
            modely.add(LSTM(64, input_shape=input_shape, return_sequences=True))
            modely.add(Dropout(0.3))
            modely.add(LSTM(64, input_shape=input_shape, return_sequences=False))
            modely.add(Dropout(0.3))
            modely.add(Flatten())
            modely.add(Dense(16,activation='elu'))
            modely.add(Dropout(0.3))
            modely.add(Flatten())
            dummy_input_x = Input(shape=(4,4))
            dummy_input_y = Input(shape=(4,4))

        elif self.model_type == "LSTM-CNN":
            # Define LSTM-CNN model
            modelx=Sequential()
            modelx.add(LSTM(64, input_shape=(4,4), return_sequences=True))
            modelx.add(Dropout(0.3))
            modelx.add(LSTM(64, input_shape=(4,4), return_sequences=False))
            modelx.add(Dropout(0.3))
            modelx.add(Flatten())
            modelx.add(Dense(16,activation='elu'))
            modelx.add(Dropout(0.3))
            modelx.add(Flatten())
            #
            modely=Sequential()
            modely.add(Conv2D(64,(2,2),padding='same',activation='relu',input_shape=(4,4,1)))
            modely.add(MaxPooling2D(2,2))  
            modely.add(Dropout(0.3))
            modely.add(Flatten())
            modely.add(Dense(16,activation='elu'))
            modely.add(Dropout(0.3))
            modely.add(Flatten())
            dummy_input_x = Input(shape=(4,4))
            dummy_input_y = Input(shape=(4,4,1))

        elif self.model_type == "CNN-CNN":
            # Define CNN-CNN model
            modelx=Sequential()
            modelx.add(Conv2D(64,(2,2),padding='same',activation='relu',input_shape=(4,4,1)))
            modelx.add(MaxPooling2D(2,2))  
            modelx.add(Dropout(0.3))
            modelx.add(Flatten())
            modelx.add(Dense(16,activation='elu'))
            modelx.add(Dropout(0.3))
            modelx.add(Flatten())
            #
            modely=Sequential()
            modely.add(Conv2D(64,(2,2),padding='same',activation='relu',input_shape=(4,4,1)))
            modely.add(MaxPooling2D(2,2))  
            modely.add(Dropout(0.3))
            modely.add(Flatten())
            modely.add(Dense(16,activation='elu'))
            modely.add(Dropout(0.3))
            modely.add(Flatten())
            dummy_input_x = Input(shape=(4,4,1))
            dummy_input_y = Input(shape=(4,4,1))
       
        # Merge the outputs of the two models
       

        dummy_output_x = modelx(dummy_input_x)
        dummy_output_y = modely(dummy_input_y)
    
        merged = Concatenate()([dummy_output_x, dummy_output_y])
        z=Dense(128,activation='relu')(merged)
        z=Dropout(0.3)(z)
        z=Dense(16,activation='relu')(z)
        z=Dense(1,activation='sigmoid')(z)


        # Create the final model
        self.model = Model(inputs=[dummy_input_x, dummy_input_y], outputs=z)
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001, decay=1e-6), metrics=['accuracy'])
        self.model.summary()


    def train_model(self, x_train_1, x_train_2, y_train_1, batch_size=25, epochs=400):
        if self.model is None:
            print("Model not built. Call build_model() first.")
            return
        if self.model_type == "CNN-LSTM":
            x_train_1=x_train_1.reshape(-1,4,4,1)
            x_train_2=x_train_2.reshape(-1,4,4)
        elif self.model_type == "LSTM-LSTM":
            x_train_1=x_train_1.reshape(-1,4,4)
            x_train_2=x_train_2.reshape(-1,4,4)
        elif self.model_type == "LSTM-CNN":
            x_train_1=x_train_1.reshape(-1,4,4)
            x_train_2=x_train_2.reshape(-1,4,4,1)
        elif self.model_type == "CNN-CNN":
            x_train_1=x_train_1.reshape(-1,4,4,1)
            x_train_2=x_train_2.reshape(-1,4,4,1)

        self.model.fit([x_train_1, x_train_2], y_train_1, batch_size=25, epochs=10, verbose=1)
        self.model.save("trainedPHNN.keras")
   