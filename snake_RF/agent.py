import keras.api._v2.keras as K
import tensorflow as tf
import numpy as np
from environments_fully_observable import *
from environments_partially_observable import *

def value_function(input_dim):
    X_input=tf.keras.Input(input_dim)
    X=tf.keras.layers.Dense(64)(X_input)
    X=tf.keras.layers.BatchNormalization()(X)
    X=tf.keras.layers.Activation(tf.nn.tanh)(X)
    X=tf.keras.layers.Dense(64)
    X=tf.keras.layers.BatchNormalization()(X)
    X=tf.keras.layers.Activation(tf.nn.tanh)(X)
    X_final=tf.keras.layers.Dense(1, activation=tf.nn.softmax,
                                  kernel_initializer=tf.initializers.RandomNormal(stddev=0.005),
                                  bias_initializer=tf.initializers.RandomNormal(stddev=0.005))(X)
    model=tf.keras.Model(X_input, X_final)
    return model 

def actor_function(input_dim):
    X_input=tf.keras.Input(input_dim)
    X=tf.keras.layers.Dense(64)(X_input)
    X=tf.keras.layers.BatchNormalization()(X)
    X=tf.keras.layers.Activation(tf.nn.tanh)(X)
    X=tf.keras.layers.Dense(64)
    X=tf.keras.layers.BatchNormalization()(X)
    X=tf.keras.layers.Activation(tf.nn.tanh)(X)
    X_final=tf.keras.layers.Dense(5, activation=tf.nn.softmax,
                                  kernel_initializer=tf.initializers.RandomNormal(stddev=0.005),
                                  bias_initializer=tf.initializers.RandomNormal(stddev=0.005))(X)
    model=tf.keras.Model(X_input, X_final)
    return model
    
def critic_function(input_dim):
    X_input=tf.keras.Input(input_dim)
    X=tf.keras.layers.Dense(64)(X_input)
    X=tf.keras.layers.BatchNormalization()(X)
    X=tf.keras.layers.Activation(tf.nn.tanh)(X)
    X=tf.keras.layers.Dense(64)(X)
    X=tf.keras.layers.BatchNormalization()(X)
    X=tf.keras.layers.Activation(tf.nn.tanh)(X)
    X_final=tf.keras.layers.Dense(1, activation="linear")(X)

    model=tf.keras.Model(X_input, X_final)
    return model