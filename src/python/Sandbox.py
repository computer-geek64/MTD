#!/usr/bin/python3
# Sandbox.py
# Ashish D'Souza
# November 30th, 2018

import tensorflow as tf
import numpy as np


x_training_data = tf.placeholder(dtype=tf.float32, shape=[None, 1])
y_training_data = tf.placeholder(dtype=tf.float32, shape=[None, 1])

hidden_layer = tf.layers.dense(x_training_data, 1, activation=tf.nn.leaky_relu)
output = tf.layers.dense(hidden_layer, 1)

loss = tf.divide(tf.reduce_sum(tf.square(tf.subtract(y_training_data, output))), 1)

optimizer = tf.train.ProximalAdagradOptimizer(0.001)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

for i in range(10000):
    sess.run(train, feed_dict={x_training_data: [[0], [1], [2], [3], [4]], y_training_data: [[3], [4], [5], [6], [7]]})

training_loss = sess.run(loss, feed_dict={x_training_data: [[0], [1], [2], [3], [4]], y_training_data: [[3], [4], [5], [6], [7]]})
prediction = sess.run(output, feed_dict={x_training_data: [[0], [1], [2], [3], [4]]})
print(prediction)
