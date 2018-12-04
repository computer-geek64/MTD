#!/usr/bin/python3
# DeepLearning.py
# Ashish D'Souza
# November 30th, 2018

import tensorflow as tf
import numpy as np


x_training_data = tf.placeholder(dtype=tf.float32, shape=[None, 1])
y_training_data = tf.placeholder(dtype=tf.float32, shape=[None, 1])

hidden_layer_weights = tf.Variable()
hidden_layer_bias = tf.Variable()

hidden_layer = tf.nn.relu(tf.add(tf.matmul(x_training_data, hidden_layer_weights), hidden_layer_bias))
