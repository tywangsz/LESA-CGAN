from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv1D, Conv2D, Flatten, GRU
from tensorflow.keras.layers import Reshape, Conv2DTranspose, UpSampling1D
from tensorflow.keras.layers import LeakyReLU, ReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate, Flatten, LSTM, Bidirectional
from tensorflow.keras.models import Model
from tensorflow import reduce_sum,squeeze
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from load_preprocess_data import google_data_loading
from Transformer_decoder import Decoder
import tensorflow as tf


seq_length = 1
dataX, X_train, scaler = google_data_loading(seq_length)

features_n = 81

noise_dim = seq_length * features_n
SHAPE = (seq_length, features_n)
hidden_dim = features_n * 4

def generator(inputs,
              activation='sigmoid',
              labels=None,
              codes=None):
    #Build a Generator Model


    if codes is not None:

        inputs = [inputs, codes]
        x=inputs[0]
        y=inputs[1]
        # noise inputs + conditional codes

    x = Reshape((SHAPE[0], (SHAPE[1]-1)))(x)
    y = Reshape((SHAPE[0], (SHAPE[1]-1)))(y)
    y, y_1 =  Decoder(1, 1, 1, 1, 80, 80)(y, x, training= False, look_ahead_mark = None, padding_mark = None)
    x = BatchNormalization()(y)
    x = reduce_sum(x, axis=0)
    x = Reshape((SHAPE[0], (SHAPE[1] - 1)))(x)
    x = Dense(80)(x)
    if activation is not None:
        x = Activation(activation)(x)

    # generator output is the synthesized data x
    return Model(inputs, x, name='gen1')



def discriminator(inputs,
                  activation='sigmoid',
                  num_labels=None,
                  num_codes=None):
    #Build a Discriminator Model

    ints = 4
    x = inputs
    x = GRU((SHAPE[1]-1) * SHAPE[0], return_sequences=False, return_state=False, unroll=True, activation="relu")(x)
    x = Reshape((ints, -1))(x)
    x = Conv1D(16, 3, 2, "same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv1D(32, 3, 2, "same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv1D(64, 3, 2, "same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv1D(128, 3, 1, "same")(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    outputs = Dense(1)(x)

    if num_codes is not None:

        z0_recon = Dense(num_codes)(x)
        z0_recon = Activation('tanh', name='z0')(z0_recon)
        outputs = [outputs, z0_recon]

    return Model(inputs, outputs, name='discriminator')




