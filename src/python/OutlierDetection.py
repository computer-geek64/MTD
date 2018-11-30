#!/usr/bin/python3
# OutlierDetection.py
# Ashish D'Souza
# November 26th, 2018

import numpy as np
import tensorflow as tf
import Data


# Detects outliers using the statistical definition of an outlier (IQR)
def statistical_outlier_detection(data, vector_index=0):
    if len(np.shape(data)) == 1:
        data = np.array([data])
    q25 = np.percentile(np.array(data[vector_index]), 25)
    q75 = np.percentile(np.array(data[vector_index]), 75)
    iqr = q75 - q25
    threshold = iqr * 1.5
    median = Data.median(data[vector_index])

    outliers = []
    for observation in range(len(data[vector_index])):
        if median - threshold > data[vector_index][observation] or median + threshold < data[vector_index][observation]:
            outliers.append(observation)
    return outliers


# Detects outliers using the standard deviation of a sample
def standard_deviation_outlier_detection(data, std=1, vector_index=0):
    if len(np.shape(data)) == 1:
        data = np.array([data])
    standard_deviation = Data.sample_standard_deviation(data[vector_index])
    mean = Data.mean(data[vector_index])

    outliers = []
    for observation in range(len(data[vector_index])):
        if mean - std * standard_deviation > data[vector_index][observation] or mean + std * standard_deviation < data[vector_index][observation]:
            outliers.append(observation)
    return outliers


# Detects outliers based on k-Nearest Neighbor machine learning algorithm with Euclidean distance formula
def knn_outlier_detection(data, k=3):
    training_data = tf.constant(data, dtype=tf.float32)  # Entire training dataset
    test_point = tf.placeholder(dtype=tf.float32)  # Current observation in training dataset

    squared_difference = tf.square(tf.subtract(training_data, test_point))
    euclidean_distance = tf.sqrt(tf.reduce_sum(squared_difference, axis=1))  # Euclidean distance

    knn_distances = tf.negative(tf.nn.top_k(tf.negative(euclidean_distance), k + 1).values)[1:]
    mean_knn_distance = tf.reduce_mean(knn_distances)  # Average of the k smallest distances

    init = tf.global_variables_initializer()
    sess = tf.Session()

    sess.run(init)

    mean_knn_distances = [sess.run(mean_knn_distance, feed_dict={test_point: data[observation]}) for observation in range(len(data))]  # List of average k-NN distances
    return statistical_outlier_detection(mean_knn_distances)
