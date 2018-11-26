# Sandbox.py
# Ashish D'Souza
# November 19th, 2018

import tensorflow as tf
import numpy as np


k = 2
data = [
    [0, 3, 5, 15],
    [1, 6, 4, 17],
    [2, 9, 3, 19],
    [3, 12, 2, 21],
    [4, 15, 1, 23]
]

training_data = tf.placeholder(dtype=tf.float32)
testing_data = tf.placeholder(dtype=tf.float32)

distances = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(training_data, testing_data)), axis=1))

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)
for observation in range(len(data)):
    distance = sess.run(distances, feed_dict={training_data: data, testing_data: data[observation]})
    print(distance)
    lowest_k_distances = sess.run(tf.nn.top_k(tf.negative(distance), k + 1))[1][1:]
    print(lowest_k_distances)
