#!/usr/bin/python3
# Main.py
# Ashish D'Souza
# December 5th, 2018

import Data
import OutlierDetection
import DeepLearning
import tensorflow as tf
import numpy as np
from datetime import datetime
import os


parameters = ["Temp", "NO2", "NOX", "NOY", "RH", "Wind Speed V", "SO2 Trace Level", "Ozone"]

soda = Data.SODA("data.delaware.gov", "2bb6-s69t")
print(soda.get_columns())
where_query = soda.format_where_query(["countycode=\"3\"", "NOT stt_datastatuscodetext=\"Down\"", "NOT stt_datastatuscodetext=\"NoData\"", "NOT stt_datastatuscodetext=\"InVld\"", "sta_stationname=\"Martin Luther King\"", "date_time>\"2017\"", "date_time<\"2018\""])
results = soda.download(where=where_query, order="date_time ASC", limit=1000000)
print(len(results))
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
    if len([parameters[i] for i in range(len(parameters)) if parameters[i] in list(data_dict[date_time].keys())]) == len(parameters):
        data.append([data_dict[date_time][monitor_name] for monitor_name in parameters])
        date = datetime.strptime(date_time.split(".")[0], "%Y-%m-%dT%H:%M:%S")
        data[-1].insert(-2, date.hour)
        data[-1].insert(-2, date.month)
    else:
        data_dict.pop(date_time)

if len(max(data)) != len(min(data)):
    print("Length of data matrix is inconsistent")
    exit(0)

# k = int(round(len(data) * 0.01))
k = 3
knn_logdir = "./tensorboard/knn_outlier_detection/" + str(int(datetime.now().timestamp())) + "/" + str(k)
outliers = OutlierDetection.knn_outlier_detection(data, k, knn_logdir)

print("All parameters: " + str(all_parameters))
print("Dataset size: " + str(len(results)) + " --> " + str(original_length) + " --> " + str(len(data)))
print("k: " + str(k))
print("Outliers: " + str(len(outliers)) + ", " + str(round(len(outliers) / len(data) * 100, 2)) + "%")

corrected_data = [data[i] for i in range(len(data)) if i not in outliers]

y_data = np.array(corrected_data)[:, -1:]
x_data = np.array(corrected_data)[:, :-1]


#layers = [9, 6, 1]
#activation_functions = [None, tf.nn.leaky_relu, None]
#optimizer = tf.train.AdagradOptimizer
#learning_rate = 0.1
#iterations = 100000
#dnn_logdir = "./tensorboard/dnn/" + str(int(datetime.now().timestamp()))
#dnn = DeepLearning.DeepNeuralNetwork(layers, activation_functions, optimizer, learning_rate, dnn_logdir)
#dnn.restore("./models/" + "-".join(list(map(str, layers))) + "_" + activation_functions[1].__name__ + "/" + optimizer.__name__ + "_" + str(learning_rate) + "_" + str(iterations) + "/model")
#print(dnn.predict(x_data[:1]))
#print(y_data[:1])
#average = Data.mean(y_data, vector_index=0)
#loss = dnn.test(x_data, y_data)
#print("Average: " + str(average))
#print("Percentage: " + str(round(loss / average * 100, 2)) + "%")
#exit(0)

optimal_dnn = float("inf")
for i in range(3):
    layers = [len(x_data[0]), 6, 4, 1]
    activation_functions = [None, tf.nn.leaky_relu, tf.nn.leaky_relu, None]
    optimizer = tf.train.AdagradOptimizer
    learning_rate = 0.1
    iterations = 100000
    dnn_logdir = "./tensorboard/dnn/" + str(int(datetime.now().timestamp())) + "/" + "-".join(list(map(str, layers))) + "_" + activation_functions[1].__name__ + "/" + optimizer.__name__ + "_" + str(learning_rate) + "_" + str(iterations)

    dnn = DeepLearning.DeepNeuralNetwork(layers, activation_functions, optimizer, learning_rate, dnn_logdir)
    loss = dnn.train(x_data, y_data, iterations)
    if optimal_dnn > loss:
        optimal_dnn = loss
        save_path = "./models/" + "-".join(list(map(str, layers))) + "_" + activation_functions[1].__name__ + "/" + optimizer.__name__ + "_" + str(learning_rate) + "_" + str(iterations) + "/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        dnn.save(os.path.join(save_path, "model"))
    average = Data.mean(y_data, vector_index=0)
    print("Loss: " + str(loss))
    print("Average: " + str(average))
    print("Percentage: " + str(round(loss / average * 100, 2)) + "%")
