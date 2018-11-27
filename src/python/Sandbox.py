# Sandbox.py
# Ashish D'Souza
# November 26th, 2018

import tensorflow as tf
import numpy as np
import Data


k_percentage = 0.01
data = [
    [2, 5],
    [8, 9],
    [1, 3],
    [4, 4],
    [9, 6],
    [11, 13],
    [5, 10],
    [8, 8],
    [9, 15],
    [1, 9],
    [1, 19]
]

training_data = tf.placeholder(dtype=tf.float32)
testing_data = tf.placeholder(dtype=tf.float32)
distance_placeholder = tf.placeholder(dtype=tf.float32)

k = int(k_percentage * len(data)) if len(data) >= 300 else 3

squared_differences = tf.square(tf.subtract(training_data, testing_data))
euclidean_distances = tf.sqrt(tf.reduce_sum(squared_differences, axis=1))

knn_distances = tf.negative(tf.nn.top_k(tf.negative(distance_placeholder), k + 1).values)[1:]

knn_distance_mean = tf.reduce_mean(knn_distances)

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

knn_distance_means = []
for observation in range(len(data)):
    distance = sess.run(euclidean_distances, feed_dict={training_data: data, testing_data: data[observation]})
    knn_distance_means.append(sess.run(knn_distance_mean, feed_dict={distance_placeholder: distance}))
print(Data.iqr(knn_distance_means) * 1.5)
print(Data.median(knn_distance_means))
print(knn_distance_means)
print([x for x in knn_distance_means if x < Data.iqr(knn_distance_means) * 1.5 + Data.median(knn_distance_means)])
