# Sandbox.py
# Ashish D'Souza
# November 26th, 2018

import tensorflow as tf
import numpy as np
import Data
import OutlierDetection


k_percentage = 0.01
data = [
    [1, 2],
    [1, 2.1],
    [1.1, 2.1],
    [1.1, 2],
    [3, 5],
    [3, 5.1],
    [3.1, 5],
    [3.1, 5.1]
]

print("Outliers: " + str(OutlierDetection.knn_outlier_detection(data)))
