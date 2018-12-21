#!/usr/bin/python3
# Main.py
# Ashish D'Souza
# @computer-geek64
# December 21st, 2018

import Data
import DeepLearning
from DownloadData import DownloadData
import tensorflow as tf
import numpy as np
from datetime import datetime
import os
import sys


parameters = ["Temp", "NO2", "NOX", "NOY", "RH", "Wind Speed V", "SO2 Trace Level", "Ozone"]

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

download_data = DownloadData(parameters)
results = download_data.download_training_data()
data = download_data.format_data(results)
x_data, y_data = download_data.remove_outliers(data, 3)

results_test = download_data.download_testing_data()
data_test = download_data.format_data(results_test)
x_data_test, y_data_test = download_data.remove_outliers(data_test, 3)

average_train = Data.mean([data[observation][-1] for observation in range(len(np.array(data)[:, -3])) if
                           8 <= data[observation][-3] <= 18], vector_index=0)
average_test = Data.mean([data_test[observation][-1] for observation in range(len(np.array(data_test)[:, -3])) if
                          8 <= data_test[observation][-3] <= 18], vector_index=0)

optimal_dnn = float("inf")
for i in range(5):
    layers = [len(x_data[0]), 6, 4, 1]
    activation_functions = [None, tf.nn.leaky_relu, tf.nn.leaky_relu, None]
    optimizer = tf.train.AdagradOptimizer
    learning_rate = 0.1
    iterations = 100000
    dnn_logdir = "./tensorboard/dnn/" + str(int(datetime.now().timestamp())) + "/" + "-".join(list(map(str, layers))) + "_" + activation_functions[1].__name__ + "/" + optimizer.__name__ + "_" + str(learning_rate) + "_" + str(iterations)

    dnn = DeepLearning.DeepNeuralNetwork(layers, activation_functions, optimizer, learning_rate, dnn_logdir)
    if "train" in sys.argv:
        dnn.restore("./models/" + "-".join(list(map(str, layers))) + "_" + activation_functions[1].__name__ + "/" + optimizer.__name__ + "_" + str(learning_rate) + "_" + str(iterations) + "/model")
        print(dnn.test(x_data, y_data))
        print(average_train)
    if "eval" in sys.argv:
        dnn.restore("./models/" + "-".join(list(map(str, layers))) + "_" + activation_functions[1].__name__ + "/" + optimizer.__name__ + "_" + str(learning_rate) + "_" + str(iterations) + "/model")
        print(dnn.test(x_data_test, y_data_test))
        print(average_test)
        exit(0)
    elif "predict" in sys.argv:
        dnn.restore("./models/" + "-".join(list(map(str, layers))) + "_" + activation_functions[1].__name__ + "/" + optimizer.__name__ + "_" + str(learning_rate) + "_" + str(iterations) + "/model")
        print(dnn.predict(x_data_test[int(sys.argv[2]):int(sys.argv[2]) + 1]))
        print(y_data_test[int(sys.argv[2]):int(sys.argv[2]) + 1])
        exit(0)

    loss_train = dnn.train(x_data, y_data, iterations)
    loss_test = dnn.test(x_data_test, y_data_test)
    if optimal_dnn > loss_test:
        optimal_dnn = loss_test
        save_path = "./models/" + "-".join(list(map(str, layers))) + "_" + activation_functions[1].__name__ + "/" + optimizer.__name__ + "_" + str(learning_rate) + "_" + str(iterations) + "/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        dnn.save(os.path.join(save_path, "model"))
    print("Training:")
    print("\tLoss: " + str(loss_train))
    print("\tAverage: " + str(average_train))
    print("\tPercentage: " + str(round(loss_train / average_train * 100, 2)) + "%")

    print("Testing:")
    print("\t" + str(dnn.predict(x_data_test[:1])))
    print("\t" + str(y_data_test[:1]))
    print("\tAverage: " + str(average_test))
    print("\tPercentage: " + str(round(loss_test / average_test * 100, 2)) + "%")
    print("\tLoss: " + str(loss_test))
