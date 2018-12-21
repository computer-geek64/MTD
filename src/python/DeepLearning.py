#!/usr/bin/python3
# DeepLearning.py
# Ashish D'Souza
# December 5th, 2018

import tensorflow as tf
from datetime import datetime


class DeepNeuralNetwork:
    def __init__(self, layers, activation_functions, tensorflow_optimizer=tf.train.AdagradOptimizer, learning_rate=0.1, logdir="./tensorboard/dnn/"):
        self.reset()
        if len(activation_functions) < len(layers):
            activation_functions.insert(0, None)
        self.y_training_data = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="y_training_data")
        with tf.name_scope("deep_neural_network"):
            with tf.name_scope("input_layer"):
                self.x_training_data = tf.placeholder(dtype=tf.float32, shape=[None, layers[0]], name="x_training_data")

            self.hidden_layers = []
            # with tf.name_scope("hidden_layer"):
            self.hidden_layers.append(tf.layers.dense(self.x_training_data, layers[1], activation=activation_functions[1], name="hidden_layer"))
            for i in range(2, len(layers) - 1):
                #with tf.name_scope("hidden_layer" + str(i)):
                self.hidden_layers.append(tf.layers.dense(self.hidden_layers[-1], layers[i], activation=activation_functions[i], name="hidden_layer" + str(i)))

            # with tf.name_scope("output_layer"):
            self.output_layer = tf.layers.dense(self.hidden_layers[-1], layers[-1], activation=activation_functions[-1], name="output_layer")

        with tf.name_scope("loss"):
            self.loss = tf.divide(tf.reduce_sum(tf.abs(tf.subtract(self.y_training_data, self.output_layer))), tf.cast(tf.shape(self.y_training_data)[0], dtype=tf.float32))
            tf.summary.scalar(name="loss_summary", tensor=self.loss)

        with tf.name_scope("training"):
            optimizer = tensorflow_optimizer(learning_rate=learning_rate, name="optimizer")
            self.training = optimizer.minimize(self.loss)

        self.summaries = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

        init = tf.global_variables_initializer()
        self.sess = tf.Session()

        self.sess.run(init)

    def train(self, x_data, y_data, training_iterations):
        for training_iteration in range(training_iterations):
            self.sess.run(self.training, feed_dict={self.x_training_data: x_data, self.y_training_data: y_data})
            self.writer.add_summary(self.sess.run(self.summaries, feed_dict={self.x_training_data: x_data, self.y_training_data: y_data}), training_iteration)

        return self.sess.run(self.loss, feed_dict={self.x_training_data: x_data, self.y_training_data: y_data})

    def test(self, x_data, y_data):
        test_loss = tf.divide(tf.reduce_sum(tf.abs(tf.subtract(tf.constant(y_data, dtype=tf.float32), self.output_layer))), tf.constant(len(y_data), dtype=tf.float32))
        return self.sess.run(test_loss, feed_dict={self.x_training_data: x_data})

    def predict(self, x_data):
        return self.sess.run(self.output_layer, feed_dict={self.x_training_data: x_data})

    def save(self, filepath_prefix="./models/model"):
        saver = tf.train.Saver()
        return saver.save(self.sess, filepath_prefix)

    def restore(self, filepath_prefix="./models/model"):
        saver = tf.train.Saver()
        saver.restore(self.sess, filepath_prefix)
        return filepath_prefix

    def reset(self):
        tf.reset_default_graph()
