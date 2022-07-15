import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

print(tf.__version__)


coefficient = np.array([[1.], [-10.], [25.]])
w = tf.Variable(0., dtype=tf.float32)

x = tf.placeholder(tf.float32, [3, 1])

#cost = tf.add(tf.add(w**2., tf.multiply(-10., w)), 25.)
#cost = w**2 - 10*w +25

cost = x[0][0]*w**2 - x[1][0]*w + x[2][0]

train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()

session = tf.Session()
session.run(init)
print(session.run(w))


#session.run(train)
#print(session.run(w))

for i in range(1000):
    session.run(train, feed_dict={x: coefficient})

print(session.run(w))




a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a, b)
print(c)
with  tf.Session() as sess:
    print(sess.run(c))


y_hat = tf.constant(36, name = 'y_hat')
y = tf.constant(39, name = 'y')
loss = tf.Variable((y - y_hat) ** 2, name='loss')

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    print(session.run(loss))







