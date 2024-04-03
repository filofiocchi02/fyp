import os
from math import sqrt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Flatten, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow_probability import distributions as tfd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils import *

def create_model_mlp_non_probabilistic(X_train, seed):
    keras.utils.set_random_seed(seed)
    
    inputs = Input(shape=(X_train.shape[1],))
    hidden = Dense(80, activation="relu")(inputs)
    hidden = Dense(50, activation="relu")(hidden)
    hidden = Dense(20, activation="relu")(hidden)
    output = Dense(1, activation="linear")(hidden) 

    model_mlp_non_probabilistic = Model(inputs=inputs, outputs=output)
    model_mlp_non_probabilistic.compile(Adam(learning_rate=0.001), loss='mean_squared_error')
    
    return model_mlp_non_probabilistic

def create_model_bnn_non_probabilistic_flipout(X_train, seed):
    keras.utils.set_random_seed(seed)
    
    inputs = Input(shape=(X_train.shape[1],))
    
    hidden1 = tfp.layers.DenseFlipout(
            units=30,
            kernel_prior_fn = tfp.layers.default_mean_field_normal_fn(),
            kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn() ,
            activation = "relu"
        )(inputs)
    
    hidden2 = tfp.layers.DenseFlipout(
            units=30,
            kernel_prior_fn = tfp.layers.default_mean_field_normal_fn(),
            kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn() ,
            activation = "relu"
        )(hidden1)
    
    hidden3 = tfp.layers.DenseFlipout(
            units=20,
            kernel_prior_fn = tfp.layers.default_mean_field_normal_fn(),
            kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn() ,
            activation = "relu"
        )(hidden2)
    
    output = tfp.layers.DenseFlipout(
            units=1,
            kernel_prior_fn = tfp.layers.default_mean_field_normal_fn(),
            kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn() ,
            activation = "relu"
        )(hidden2)
    
    layers = [hidden1, hidden2, hidden3, output]

    model_mlp_non_probabilistic_bnn = Model(inputs=inputs, outputs=output)
    model_mlp_non_probabilistic_bnn.compile(Adam(learning_rate=0.001), loss='mean_squared_error')

    return model_mlp_non_probabilistic_bnn, layers


def create_model_bnn(X_train, seed):
    keras.utils.set_random_seed(seed) 
    
    kl_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (X_train.shape[0] * 1.0)
        
    inputs = Input(shape=(X_train.shape[1],))
    
    hidden1 = tfp.layers.DenseFlipout(
            units=80,
            kernel_prior_fn = tfp.layers.default_mean_field_normal_fn(),
            kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn() ,
            kernel_divergence_fn = kl_divergence_fn,
            activation = "relu"
        )(inputs)
    
    hidden2 = tfp.layers.DenseFlipout(
            units=50,
            kernel_prior_fn = tfp.layers.default_mean_field_normal_fn(),
            kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn() ,
            kernel_divergence_fn = kl_divergence_fn,
            activation = "relu"
        )(hidden1)
    
    hidden3 = tfp.layers.DenseFlipout(
            units=20,
            kernel_prior_fn = tfp.layers.default_mean_field_normal_fn(),
            kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn() ,
            kernel_divergence_fn = kl_divergence_fn,
            activation = "relu"
        )(hidden2)
    
    # Output Univariate Normal Probabilistic Layer
    dist_params = Dense(2)(hidden3)
    dist = tfp.layers.DistributionLambda(normal_softplus)(dist_params)

    model_bnn = Model(inputs=inputs, outputs=dist)
    model_bnn.compile(Adam(learning_rate=0.001), loss=NLL)

    return model_bnn


def create_model_mlp_gaussian_separate(X_train, seed):
    keras.utils.set_random_seed(seed)
    
    inputs = Input(shape=(X_train.shape[1],))
    mean_h1 = Dense(80, activation="relu")(inputs)
    variance_h1 = Dense(80, activation="relu")(inputs)
    
    mean_h2 = Dense(50, activation="relu")(mean_h1)
    variance_h2 = Dense(50, activation="relu")(variance_h1)
    
    mean_h3 = Dense(20, activation="relu")(mean_h2)
    variance_h3 = Dense(20, activation="relu")(variance_h2)
    
    mean_h4 = Dense(20, activation="relu")(mean_h3)
    variance_h4 = Dense(20, activation="relu")(variance_h3)
    
    mean_out = Dense(1)(mean_h4)
    variance_out = Dense(1)(variance_h4)
    
    params = Concatenate()([mean_out, variance_out])
    
    dist = tfp.layers.DistributionLambda(normal_softplus)(params) 

    model_mlp_gaussian = Model(inputs=inputs, outputs=dist)
    model_mlp_gaussian.compile(Adam(learning_rate=0.001), loss=NLL)

    return model_mlp_gaussian


def create_model_mlp_gaussian_joint(X_train, seed):
    keras.utils.set_random_seed(seed)
    
    inputs = Input(shape=(X_train.shape[1],))
    
    hidden1 = Dense(100, activation="relu")(inputs)
    hidden2 = Dense(80, activation="relu")(hidden1)
    hidden3 = Dense(40, activation="relu")(hidden2)
    
    mean_h1 = Dense(20, activation="relu")(hidden3)
    mean_out = Dense(1)(mean_h1)
    
    variance_h1 = Dense(20, activation="relu")(hidden3)
    variance_out = Dense(1)(variance_h1)
    
    params = Dense(2)(Concatenate()([mean_out, variance_out]))

    
    dist = tfp.layers.DistributionLambda(normal_softplus)(params) 

    model_mlp_gaussian = Model(inputs=inputs, outputs=dist)
    model_mlp_gaussian.compile(Adam(learning_rate=0.001), loss=NLL)

    return model_mlp_gaussian


def create_model_cnn_gaussian(X_train, seed):
    keras.utils.set_random_seed(seed)
    
    inputs = Input(shape=(X_train.shape[1], 1))
    conv1d_layer = Conv1D(filters=32, kernel_size=5, activation='relu')(inputs)
    maxpooling_layer = MaxPooling1D(pool_size=2)(conv1d_layer)
    
    flatten_layer = Flatten()(maxpooling_layer)
 
    hidden1 = Dense(50, activation="relu")(flatten_layer)
    hidden2 = Dense(50, activation="relu")(hidden1)
    hidden3 = Dense(20, activation="relu")(hidden2)
    
    params = Dense(2)(hidden3)
    dist = tfp.layers.DistributionLambda(normal_softplus)(params) 

    model_cnn_gaussian = Model(inputs=inputs, outputs=dist)
    model_cnn_gaussian.compile(Adam(learning_rate=0.001), loss=NLL)
    
    return model_cnn_gaussian


def create_model_multivariate_gaussian_only_diagonal(d, input_size, seed):
    keras.utils.set_random_seed(seed)
    
    inputs = Input(shape=input_size)
    
    outs = []

    for i in range(d):
        
        h_means1 = Dense(50, activation='relu')(inputs)
        h_cov1 = Dense(50, activation='relu')(inputs)
    
        h_means2 = Dense(30, activation='relu')(h_means1)
        h_cov2 = Dense(30, activation='relu')(h_cov1)
    
        h_means3 = Dense(20, activation='relu')(h_means2)
        h_cov3 = Dense(20, activation='relu')(h_cov2)
        
        out = Dense(2)(Concatenate()([h_means3, h_cov3]))
        outs.append(out)

    concatenated_outputs = Concatenate()(outs)
    
    # Define the distribution layer
    distribution_layer = tfp.layers.DistributionLambda(
        lambda t: multivariate_diagonal_normal_softplus(t[:, 0::2], t[:, 1::2], d)
    )
    
    model = Model(inputs=inputs, outputs=distribution_layer(concatenated_outputs),
                  name="multivariate_gaussian_with_covariance")
    
    model.compile(Adam(learning_rate=0.01),
                  loss=NLL)
   
    return model


def create_model_multivariate_gaussian_only_diagonal_common(d, input_size, seed):
    keras.utils.set_random_seed(seed)
    
    inputs = Input(shape=input_size)
    
    outs = []
        
    h1 = Dense(100, activation='relu')(inputs)
    h2 = Dense(50, activation='relu')(h1)
    h3 = Dense(30, activation='relu')(h2)
    
    for i in range(d):
        out = Dense(2)(h3)
        outs.append(out)


    concatenated_outputs = Concatenate()(outs)
    
    # Define the distribution layer
    distribution_layer = tfp.layers.DistributionLambda(
        lambda t: multivariate_diagonal_normal_softplus(t[:, 0::2], t[:, 1::2], d)
    )
    
    model = Model(inputs=inputs, outputs=distribution_layer(concatenated_outputs),
                  name="multivariate_gaussian_with_covariance")
    
    model.compile(Adam(learning_rate=0.001),
                  loss=NLL)
   
    return model


def create_model_multivariate_gaussian_with_covariance(d, input_size, seed):
    keras.utils.set_random_seed(seed)
    
    inputs = Input(shape=input_size)

    h1_mean = BatchNormalization()(Dense(150, activation='relu')(inputs))
    h2_mean = BatchNormalization()(Dense(70, activation='relu')(h1_mean))
    h3_mean = BatchNormalization()(Dense(50, activation='relu')(h2_mean))
    
    h1_cov = BatchNormalization()(Dense(150, activation='relu')(inputs))
    h2_cov = BatchNormalization()(Dense(70, activation='relu')(h1_cov))
    h3_cov = BatchNormalization()(Dense(50, activation='relu')(h2_cov))
    
    out_mean = Dense(d, activation='linear')(h3_mean)
    out_cov = Dense(tfp.layers.MultivariateNormalTriL.params_size(d) - d)(h3_cov)
    
    # Concatenate mean and lower triangular part of the covariance matrix
    concatenated_outputs = Concatenate()([out_mean, out_cov])
    
    # Define the distribution layer
    distribution_layer = tfp.layers.DistributionLambda(
        lambda t: multivariate_covariance_normal_softplus(t[:, :d], t[:, d:], d)
    )
    
    model = Model(inputs=inputs, outputs=distribution_layer(concatenated_outputs),
                  name="multivariate_gaussian_with_covariance")
    
    model.compile(Adam(learning_rate=0.001, clipnorm=100),
                  loss=NLL)
   
    return model


def create_model_mlp_gaussian_large(X_train_full, seed):
    keras.utils.set_random_seed(seed)

    inputs = Input(shape=(X_train_full.shape[1],))
    hidden1 = Dense(300, activation="relu")(inputs)
    hidden2 = Dense(200, activation="relu")(hidden1)
    hidden3 = Dense(100, activation="relu")(hidden2)

    params = Dense(2)(hidden3)

    dist = tfp.layers.DistributionLambda(normal_softplus)(params)

    model = Model(inputs=inputs, outputs=dist)
    model.compile(Adam(learning_rate=0.001), loss=NLL)

    return model


def create_model_finetune(X_train, generic_model, seed):
    keras.utils.set_random_seed(seed)

    inputs = Input(shape=(X_train.shape[1],))

    pretrained_model_layers = generic_model.layers[1:]
    l = inputs

    for layer in pretrained_model_layers:
        layer.trainable = True
        l = layer(l)

    # Note: set the learning rate to a smalller value to obtain a more accurate local minima
    model_mlp_gaussian = Model(inputs=inputs, outputs=l)
    model_mlp_gaussian.compile(Adam(learning_rate=1e-4), loss=NLL)

    return model_mlp_gaussian