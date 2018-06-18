#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 16:44:04 2018

@author: fuzijie
"""

import tensorflow as tf

x = tf.Variable(3, name = "x")
y = tf.Variable(4, name = "y")
f = x*x*y+y+2

sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
#print(result)
sess.close()

with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
# automatically closed

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
init.run()
result = f.eval()
#print(result)
sess.close()

x1 = tf.Variable(1)
x1.graph is tf.get_default_graph()

graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)

x2.graph is graph
x2.graph is tf.get_default_graph()

import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype = tf.float32, name= "X")
y = tf.constant(housing.target.reshape(-1, 1), dtype = tf.float32, name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()

#print(theta_value)


n_epochs = 1000
learning_rate = 0.01

scaled_housing_data_plus_bias = (housing_data_plus_bias - np.mean(housing_data_plus_bias))/np.std(housing_data_plus_bias)
X = tf.constant(scaled_housing_data_plus_bias, dtype = tf.float32, name= "X")
y = tf.constant(housing.target.reshape(-1, 1), dtype = tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
#gradients = 2/m * tf.matmul(tf.transpose(X), error)
#gradients = tf.gradients(mse, [theta])[0]
#training_op = tf.assign(theta, theta - learning_rate * gradients)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
#saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
#            save_path = saver.save(sess, "/tmp/my_model.ckpt")
            print("Epoch",epoch,"MSE =",mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()
#    save_path = saver.save(sess, "/tmp/my_model_final.ckpt")
print(best_theta)

#with tf.Session() as sess:
#    saver.restore(sess, "/tmp/my_model_final.ckpt")
#    for epoch in range(n_epochs):
#        if epoch % 100 == 0:
#            print("Epoch",epoch,"MSE =",mse.eval())
#        sess.run(training_op)
#    best_theta = theta.eval()
#print(best_theta)


#print(B_val_1)
#print(B_val_2)

X = tf.placeholder(tf.float32, shape=(None, n+1),name="X")
y = tf.placeholder(tf.float32, shape=(None,1),name="y")
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
