#!/usr/bin/python3
# Main.py
# Ashish D'Souza
# December 5th, 2018

import Data
import OutlierDetection
import DeepLearning
import tensorflow as tf
import numpy as np


parameters = ["Temp", "NO2", "NOX", "NOY", "RH", "Wind Speed V", "Ozone"]

soda = Data.SODA("data.delaware.gov", "2bb6-s69t")
print(soda.get_columns())
where_query = soda.format_where_query(["countycode=\"3\"", "NOT stt_datastatuscodetext=\"Down\"", "NOT stt_datastatuscodetext=\"NoData\"", "NOT stt_datastatuscodetext=\"InVld\"", "sta_stationname=\"Martin Luther King\""])
results = soda.download(where=where_query, order="date_time DESC", limit=100000)

data_dict = {}
for observation in range(len(results)):
    if results[observation]["date_time"] not in data_dict.keys():
        data_dict[results[observation]["date_time"]] = {}
    data_dict[results[observation]["date_time"]][results[observation]["mot_monitorname"]] = float(results[observation]["paramvalue"])
all_parameters = []
for observation in range(len(results)):
    if results[observation]["mot_monitorname"] not in all_parameters:
        all_parameters.append(results[observation]["mot_monitorname"])
all_parameters = sorted(all_parameters)
original_length = len(data_dict)

data = []
for date_time in sorted(data_dict.keys()):
    data.append([])
    if len([parameters[i] for i in range(len(parameters)) if parameters[i] in list(data_dict[date_time].keys())]) == len(parameters):
        data[-1] = [data_dict[date_time][monitor_name] for monitor_name in parameters]
    else:
        data.pop(-1)
        data_dict.pop(date_time)

if len(max(data)) != len(min(data)):
    print("Length of data matrix is inconsistent")
    exit(0)

# k = int(round(len(data) * 0.01))
k = 3
outliers = OutlierDetection.knn_outlier_detection(data, k=k)

print("All parameters: " + str(all_parameters))
print("Dataset size: " + str(len(results)) + " --> " + str(original_length) + " --> " + str(len(data)))
print("k: " + str(k))
print("Outliers: " + str(len(outliers)) + ", " + str(round(len(outliers) / len(data) * 100, 2)) + "%")

corrected_data = [data[i] for i in range(len(data)) if i not in outliers]

y_data = np.array(corrected_data)[:, -1:]
x_data = np.array(corrected_data)[:, :-1]

dnn = DeepLearning.DeepNeuralNetwork(layers=[len(x_data[0]), int((len(x_data[0]) + 1) / 2) + 1, 1], activation_functions=[None, tf.nn.relu, None], tensorflow_optimizer=tf.train.ProximalAdagradOptimizer, learning_rate=0.01)
loss = dnn.train(x_data, y_data, 100000)
print(loss)
print(Data.mean(y_data, vector_index=0))
