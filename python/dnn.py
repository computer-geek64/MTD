# dnn.py
# Ashish D'Souza
# October 29th, 2018

import numpy as np
import tensorflow as tf


categorical_feature_a = categorical_column_with_hash_bucket(...)
categorical_feature_b = categorical_column_with_hash_bucket(...)

categorical_feature_a_emb = embedding_column(categorical_column=categorical_feature_a, ...)
categorical_feature_b_emb = embedding_column(categorical_column=categorical_feature_b, ...)

estimator = tf.estimator.DNNRegressor(feature_columns=[categorical_feature_a_emb, categorical_feature_b_emb], hidden_units=[1024, 512, 256])

