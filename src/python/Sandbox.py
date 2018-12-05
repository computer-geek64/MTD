#!/usr/bin/python3
# Sandbox.py
# Ashish D'Souza
# November 30th, 2018

import tensorflow as tf


with tf.name_scope("input_layer"):
    x_training_data = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="x_training_data")
y_training_data = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="y_training_data")

with tf.name_scope("hidden_layer"):
    hidden_layer = tf.layers.dense(x_training_data, 1, activation=tf.nn.leaky_relu)

with tf.name_scope("output_layer"):
    output = tf.layers.dense(hidden_layer, 1, name="output_layer")

with tf.name_scope("loss_function"):
    loss = tf.divide(tf.reduce_sum(tf.square(tf.subtract(y_training_data, output))), tf.constant(1, dtype=tf.float32), name="loss")

loss_summary = tf.summary.scalar(name="loss_summary", tensor=loss)

with tf.name_scope("training"):
    optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=0.1, name="optimizer")
    train = optimizer.minimize(loss, name="train")
learning_rate_summary = tf.summary.scalar(name="learning_rate_summary", tensor=optimizer._learning_rate_tensor)

summaries = tf.summary.merge_all()
writer = tf.summary.FileWriter("./graphs", tf.get_default_graph())

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

for i in range(10000):
    sess.run(train, feed_dict={x_training_data: [[0], [1], [2], [3], [4]], y_training_data: [[3], [4], [5], [6], [7]]})
    writer.add_summary(sess.run(summaries, feed_dict={x_training_data: [[0], [1], [2], [3], [4]], y_training_data: [[3], [4], [5], [6], [7]]}), i)

training_loss = sess.run(loss, feed_dict={x_training_data: [[0], [1], [2], [3], [4]], y_training_data: [[3], [4], [5], [6], [7]]})
prediction = sess.run(output, feed_dict={x_training_data: [[0], [1], [2], [3], [4]]})
print(prediction)
print(training_loss)
