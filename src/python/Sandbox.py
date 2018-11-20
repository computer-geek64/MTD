# Sandbox.py
# Ashish D'Souza
# November 19th, 2018

import tensorflow as tf


def linear_regression(data):
    variables = [tf.Variable(1.0, dtype=tf.float32)] * len(data)
    training_data = tf.placeholder(dtype=tf.float32)
    linear_model = tf.tensordot(training_data[:-1], variables, 0)

    squared_error = tf.square(tf.subtract(training_data[-1], linear_model))
    mean_squared_error = tf.divide(tf.reduce_sum(squared_error), len(squared_error.shape))

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(mean_squared_error)

    init = tf.global_variables_initializer()
    sess = tf.Session()

    sess.run(init)

    for i in range(1000):
        sess.run(train, feed_dict={training_data: data[:-1] + [1] * len(data[0]) + data[-1]})
    return sess.run([mean_squared_error, variables], feed_dict={training_data: data[:-1] + [1] * len(data[0]) + data[-1]})


print(linear_regression([[0, 1, 2, 3], [1, 2, 3, 4]]))
