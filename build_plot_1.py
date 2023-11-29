

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Activation, Dense, Input, GaussianNoise, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import concatenate, GRU, Dropout, Bidirectional
# import tensorview as tv
from gen_dis import discriminator
from gen_dis import generator
from load_preprocess_data import google_data_loading
from tensorflow.keras import Sequential
from Transformer_encoder import TransformerEncoder, EncoderLayer

import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import os
import argparse


from tensorflow.keras.layers import LSTM
import sys

sys.path.append("..")

seq_length = 1  # specifying the lenght of series

dataX, X_train, scaler = google_data_loading(seq_length)

features_n = 81

noise_dim = seq_length * features_n
SHAPE = (seq_length, features_n)
hidden_dim = features_n * 4


# from lib import gan

def build_encoder(inputs, num_labels=2,
                      feature0_dim=80 * 1):


    x, feature0 = inputs

    num_layers = 1
    d_model = 80
    num_heads = 1
    dff = 64
    window_size = 3
    dropout_rate = 0.1
    num_labels = 2
    training = 'training'

    y = TransformerEncoder(num_layers, d_model, num_heads, dff, window_size, dropout_rate)(x)

    y = GlobalMaxPooling1D()(y)
    feature0_output = Dense(feature0_dim, activation='relu')(y)



    # Encoder0 or enc0: data to feature0，
    enc0 = Model(inputs=x, outputs=feature0_output, name="encoder0")


    y1 = tf.keras.backend.expand_dims(feature0_output, axis=1)
    y2 = TransformerEncoder(num_layers, d_model, num_heads, dff, window_size, dropout_rate)(y1)
    y = Dense(num_labels)(y2)
    y3 = tf.reduce_sum(y, axis=1)
    y = Dropout(0.2)(y3)

    labels = Activation('softmax')(y)

    # Encoder1 or enc1: feature0 to class labels
    enc1 = Model(inputs=feature0_output, outputs=labels, name="encoder1")


    # return both enc0 and enc1
    return enc0, enc1


def build_generator(latent_codes, feature0_dim=80):

    # Latent codes and network parameters
    labels, z0, z1, feature0 = latent_codes


    inputs = [labels, z0, feature0]
    x = concatenate(inputs, axis=1)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(100, activation='relu')(x)
    x = BatchNormalization()(x)
    fake_feature0 = Dense(feature0_dim, activation='relu')(x)

    # gen0: classes and noise
    gen0 = Model(inputs, fake_feature0, name='gen0')

    inputs1 = [labels, z1, fake_feature0]
    x1 = concatenate(inputs1, axis=1)
    x1 = Dense(128, activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dense(100, activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    fake_feature1 = Dense(feature0_dim, activation='relu')(x1)
    # gen1: feature0 + z0 to feature1 (time series array)
    gen1 = Model(inputs1, fake_feature1, name='gen1')


    return gen0, gen1


def build_discriminator(inputs, z_dim=50):

    x = Dense(SHAPE[0] * (SHAPE[1] - 1), activation='relu')(inputs)

    x = Dense(SHAPE[0] * (SHAPE[1] - 1), activation='relu')(x)

    # first output is probability that feature0 is real
    f0_source = Dense(1)(x)
    f0_source = Activation('sigmoid',
                           name='feature1_source')(f0_source)

    z0_recon = Dense(z_dim)(x)
    z0_recon = Activation('tanh', name='z0')(z0_recon)

    discriminator_outputs = [f0_source, z0_recon]
    dis0 = Model(inputs, discriminator_outputs, name='dis0')
    return dis0


def train(models, data, params):



    enc0, enc1, gen0, gen1, dis0, dis1, adv0, adv1 = models
    # network parameters
    batch_size, train_steps, num_labels, z_dim, z_dim_g1, model_name = params
    # train dataset
    (x_train, y_train), (_, _) = data  # I can do this.
    # the generated time series array is saved every 50 steps
    save_interval = 49


    z0 = np.random.normal(scale=0.5, size=[SHAPE[0], z_dim])
    z1 = np.random.normal(scale=0.5, size=[SHAPE[0], z_dim_g1])
    noise_class = np.eye(num_labels)[np.arange(0, SHAPE[0]) % num_labels]
    noise_params = [noise_class, z0, z1]

    train_size = x_train.shape[0]
    print(model_name,
          "Labels for generated time series arrays: ",
          np.argmax(noise_class, axis=1))


    for i in range(
            200):
        dicta = {}
        rand_indexes = np.random.randint(0,
                                         train_size,
                                         size=batch_size)
        real_samples = x_train[rand_indexes]
        # real feature0 from encoder0 output
        real_feature0 = enc0.predict(real_samples)
        real_z0 = np.random.normal(scale=0.5,
                                   size=[batch_size, z_dim])
        real_labels = y_train[rand_indexes]

        # generate fake feature0 using generator0
        fake_z0 = np.random.normal(scale=0.5,
                                   size=[batch_size, z_dim])

        fake_feature0 = gen0.predict([real_labels, fake_z0, real_feature0])
        real_feature1 = enc1.predict(fake_feature0)
        feature0 = np.concatenate((real_feature0, fake_feature0))
        z0 = np.concatenate((fake_z0, fake_z0))

        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0

        # train discriminator1 to classify feature


        metrics = dis0.train_on_batch(feature0, [y, z0])
        log = "%d: [dis0_loss: %f]" % (i, metrics[0])
        dicta["dis0_loss"] = metrics[0]


        fake_z1 = np.random.normal(scale=0.5, size=[batch_size, z_dim])
        print(real_feature0.shape)
        print(fake_z1.shape)
        print(fake_feature0.shape)
        fake_samples = gen1.predict([real_feature1, fake_z1, fake_feature0])


        fake_samples = tf.keras.backend.expand_dims(fake_samples, axis=1)
        x = np.concatenate((real_samples, fake_samples))
        z1 = np.concatenate((fake_z1, fake_z1))

        # train discriminator1 to classify time series arrays
        metrics = dis1.train_on_batch(x, [y, z1])

        # log the overall loss only (use dis1.metrics_names)
        log = "%s [dis1_loss: %f]" % (log, metrics[0])
        dicta["dis1_loss"] = metrics[0]

        # adversarial training
        fake_z0 = np.random.normal(scale=0.5,
                                   size=[batch_size, z_dim])

        gen0_inputs = [real_labels, fake_z0, real_feature0]

        y = np.ones([batch_size, 1])

        metrics = adv0.train_on_batch(gen0_inputs,
                                      [y, fake_z0, real_labels])

        fmt = "%s [adv0_loss: %f, enc1_acc: %f]"
        dicta["adv0_loss"] = metrics[0]
        dicta["enc1_acc"] = metrics[6]

        # log the overall loss and classification accuracy
        log = fmt % (log, metrics[0], metrics[6])


        fake_z1 = np.random.normal(scale=0.5,
                                   size=[batch_size, z_dim])

        gen1_inputs = [real_feature1, fake_z1, fake_feature0]



        metrics = adv1.train_on_batch(gen1_inputs,
                                      [y, fake_z1, real_feature0])

        # log the overall loss only
        log = "%s [adv1_loss: %f]" % (log, metrics[0])
        dicta["adv1_loss"] = metrics[0]
        print(log)
        if (i + 1) % save_interval == 0:
            generators = (gen0, gen1)
            encoder = (enc0,enc1)
            plot_ts(generators,
                    encoder,
                    real_samples,
                    real_labels,
                    fake_z0,
                    fake_z1,
                    noise_params=noise_params,
                    show=False,
                    step=(i + 1),
                    model_name=model_name)

    # save the modelis after training generator0 & 1
    # the trained generator can be reloaded for
    # future data generation
    gen0.save(model_name + "-gen1.h5")
    gen1.save(model_name + "-gen0.h5")
    enc0.save(model_name + "-enc0.h5")

    return gen0, gen1, enc0, enc1


def plot_ts(generators,
            encoder,
            real_samples,
            real_labels,
            fake_z0,
            fake_z1,
            noise_params,
            show=False,
            step=0,
            model_name="gan"):


    gen0, gen1 = generators
    enc0,enc1 = encoder
    noise_class, z0, z1 = noise_params

    real_feature0 = enc0.predict(real_samples)
    fake_feature0 = gen0.predict([real_labels, fake_z0, real_feature0])

    real_feature1 = enc1.predict(fake_feature0)
    tss = gen1.predict([real_feature1, fake_z1, fake_feature0])


def train_encoder(model,
                  data,
                  model_name="LESA-CGAN",
                  batch_size=64):

    (x_train, y_train), (x_test, y_test) = data

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train,
              y_train,
              validation_data=(x_test, y_test),
              epochs=1,
              batch_size=batch_size)

    score = model.evaluate(x_test,
                           y_test,
                           batch_size=batch_size,
                           verbose=0)
    print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))





def build_and_train_models(train_steps=1):
    """Load the dataset, build GAN discriminator,
    generator, and adversarial models.
    Call the LESA-CGAN train routine.
    """

    dataX, _, _ = google_data_loading(seq_length)
    dataX = np.stack(dataX)

    train_n = int(len(dataX) * .70)
    X = dataX[:, :, :-1]
    y = dataX[:, -1, -1]
    x_train, y_train = X[:train_n, :, :], y[:train_n]
    x_test, y_test = X[train_n:, :, :], y[train_n:]

    # number of labels
    num_labels = len(np.unique(y_train))
    # to one-hot vector
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model_name = "LESA-CGAN"
    # network parameters
    batch_size = 64

    lr = 2e-4
    decay = 6e-8
    z_dim = 128
    z_shape = (z_dim,)

    z_dim_g1 = 80
    z_shape_g1 = (z_dim_g1,)

    feature0_dim = SHAPE[0] * (SHAPE[1] - 1)
    feature0_shape = (feature0_dim,)
    optimizer = RMSprop(lr=lr, decay=decay)


    input_shape = (feature0_dim,)
    inputs = Input(shape=input_shape, name='discriminator0_input')
    dis0 = build_discriminator(inputs, z_dim=z_dim)


    loss = ['binary_crossentropy', 'mse']
    loss_weights = [1.0, 1.0]
    dis0.compile(loss=loss,
                 loss_weights=loss_weights,
                 optimizer=optimizer,
                 metrics=['accuracy'])
    dis0.summary()



    input_shape = ((x_train.shape[1]), x_train.shape[2])
    inputs = Input(shape=input_shape, name='discriminator1_input')
    dis1 = discriminator(inputs, num_codes=z_dim)


    loss = ['binary_crossentropy', 'mse']
    loss_weights = [1.0, 10.0]
    dis1.compile(loss=loss,
                 loss_weights=loss_weights,
                 optimizer=optimizer,
                 metrics=['accuracy'])
    dis1.summary()

    # build generator models
    label_shape = (num_labels,)
    feature0 = Input(shape=feature0_shape, name='feature0_input')
    feature1 = Input(shape=feature0_shape, name='feature1_input')
    labels = Input(shape=label_shape, name='labels')
    z0 = Input(shape=z_shape, name="z0_input")
    z1 = Input(shape=z_shape, name="z1_input")
    latent_codes = (labels, z0, z1, feature0)
    gen0, gen1 = build_generator(latent_codes)
    gen0.summary()
    gen1.summary()

    # build encoder models
    input_shape = (SHAPE[0], (SHAPE[1] - 1))
    inputs = Input(shape=input_shape, name='encoder_input')
    enc0, enc1 = build_encoder((inputs, feature0), num_labels)
    enc0.summary()
    enc1.summary()
    encoder = Model(inputs, enc1(enc0(inputs)))
    encoder.summary()
    data = (x_train, y_train), (x_test, y_test)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print(x_train.shape)
    print(y_train.shape)

    train_encoder(encoder, data, model_name=model_name)


    enc1.trainable = True
    dis0.trainable = False

    gen0_inputs = [labels, z0, feature0]
    gen0_outputs = gen0(gen0_inputs)
    adv0_outputs = dis0(gen0_outputs) + [
        enc1(gen0_outputs)]


    adv0 = Model(gen0_inputs, adv0_outputs, name="adv0")


    def custom_loss1(gen0_inputs, adv0_outputs):
        custom_loss1 = (gen0_inputs - adv0_outputs) ** 2 + (gen0_inputs ** 2 - adv0_outputs ** 2)
        return custom_loss1


    adv0.compile(loss=custom_loss1,
                 loss_weights=loss_weights,
                 optimizer=optimizer,
                 metrics=['accuracy'])
    adv0.summary()

    optimizer = RMSprop(lr=lr * 0.5, decay=decay * 0.5)
    enc0.trainable = False
    dis1.trainable = False

    gen1_inputs = [labels, z1 , feature0]
    gen1_outputs = gen1(gen1_inputs)

    print(gen1_outputs)
    gen1_outputs = tf.keras.backend.expand_dims(gen1_outputs, axis=1)
    adv1_outputs = dis1(gen1_outputs) + [enc0(gen1_outputs)]

    adv1 = Model(gen1_inputs, adv1_outputs, name="adv1")

    def custom_loss2(gen1_inputs, adv1_outputs):
        custom_loss2 = (gen1_inputs - adv1_outputs) ** 2 + (gen1_inputs ** 2 - adv1_outputs ** 2)
        return custom_loss2

    loss = ['binary_crossentropy', 'mse', 'mse']
    loss_weights = [1.0, 10.0, 1.0]
    adv1.compile(loss=custom_loss2,
                 loss_weights=loss_weights,
                 optimizer=optimizer,
                 metrics=['accuracy'])
    adv1.summary()

    # train discriminator and adversarial networks
    models = (enc0, enc1, gen0, gen1, dis0, dis1, adv0, adv1)
    params = (batch_size, train_steps, num_labels, z_dim, z_dim_g1, model_name)
    gen0, gen1, enc0,enc1 = train(models, data, params)


    return gen0, gen1, enc0,enc1
