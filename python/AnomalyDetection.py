# AnomalyDetection.py
# Ashish D'Souza
# November 15th, 2018

import numpy as np
import tensorflow as tf


def get_outliers(data, elimination_criteria, vector_index=0):
    if len(np.shape(data)) == 1:
        data = np.array([data])
    return [observation for observation in data[vector_index] if not elimination_criteria(observation, data[vector_index])]


def remove_outliers(data, elimination_criteria, vector_index=0):
    if len(np.shape("data")) == 1:
        data = np.array([data])
    return [observation for observation in data[vector_index] if elimination_criteria(observation, data[vector_index])]


def k_nearest_neighbor(data, k, vector_index=0):
    if len(np.shape(data)) == 1:
        data = np.array([data])
