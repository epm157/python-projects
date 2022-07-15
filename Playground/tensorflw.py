import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)

x = 2.0
ev = tf.square(x)
x = 4.0
print(ev)

x = [[2.0]]
x = tf.square(x)
try:
    x.assign(3.0)
except:
    print('Error :(')
x = x.numpy()
print(x)

a = tf.constant([[1,2], [3, 4]])
b = tf.constant([[2, 1], [3, 4]])
c = tf.matmul(a, b)
c = c.numpy()
print(c)

d = tf.Variable(5.0)
d.assign(6.0)
d.assign_add(2)
print(d)

v = tf.Variable(5.0)
print(v.name)
print(v.value)
print(v.device)
print(v.shape)

x = tf.Variable(2.0)
with tf.GradientTape() as tape:
    y = x*x
grad = tape.gradient(y, x)
print(f'The gradient of w^2 at {x.numpy()} is {grad.numpy()}')


def sigmoid(x):
    return 1/(1 + tf.exp(-x))

x = tf.Variable(0.)
with tf.GradientTape() as tape:
    y = sigmoid(x)
grad = tape.gradient(y, x)
print('The gradient of the sigmoid function at 0.0 is ', grad.numpy())


def log(x):
    return tf.math.log(x)

x = tf.Variable(1.)
with tf.GradientTape() as t1:
    with tf.GradientTape() as t2:
        y = log(x)
    dx = t2.gradient(y, x)
dx2 = t1.gradient(dx, x)
print('The first  derivative of log at x = 1 is ', dx.numpy())
print('The second derivative of log at x = 1 is ', dx2.numpy())




from functools import reduce
numbers = [1,2,3,4,5,6]
odd_numbers = filter(lambda n: n % 2 == 1, numbers)
squared_odd_numbers = map(lambda n: n * n, odd_numbers)
total = reduce(lambda acc, n: acc + n, squared_odd_numbers)
print(total)






