#!/usr/bin/python3
# DeepLearning.py
# Ashish D'Souza
# November 30th, 2018

import tensorflow as tf


class DeepNeuralNetwork:
    def __init__(self, layers, activation_functions):

        if len(activation_functions) < len(layers):
            activation_functions.insert(0, None)
        with tf.name_scope("input_layer"):
            self.x_training_data = tf.placeholder(dtype=tf.float32, shape=[None, layers[0]], name="x_training_data")
        self.y_training_data = tf.placeholder(dtype=tf.float32, shape=[None, layers[0]], name="y_training_data")

        self.hidden_layers = []
        with tf.name_scope("hidden_layer"):
            self.hidden_layers.append(tf.layers.dense(x_training_data, layers[1], activation=activation_functions[1], name="hidden_layer"))
        for i in range(2, layers - 1):
            with tf.name_scope("hidden_layer" + str(i)):
                self.hidden_layers.append(tf.layers.dense(self.hidden_layers[-1], layers[i], activation=activation_functions[i] + str(i)))

        self.output_layer = tf.layers.dense(self.hidden_layers[-1], layers[-1], name="output_layer")


x_training_data = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="x_training_data")
y_training_data = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="y_training_data")

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(sess.run(tf.reduce_prod(tf.shape(x_training_data)), feed_dict={x_training_data: [[0], [1], [2]]}))
print(x_training_data)
print(y_training_data)
print(sess.run(tf.get_default_graph().get_tensor_by_name("x_training_data"), feed_dict={x_training_data: [[0], [1], [2]]}))
exit(0)

hidden_layer_weights = [tf.Variable(0, dtype=tf.float32) for i in range(1)]
hidden_layer_bias = [tf.Variable(0, dtype=tf.float32) for i in range(1)]

hidden_layer = tf.nn.relu(tf.add(tf.matmul(x_training_data, hidden_layer_weights), hidden_layer_bias))
