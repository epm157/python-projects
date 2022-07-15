from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import contextlib


@contextlib.contextmanager
def assert_raise(error_class):
    try:
        yield
    except error_class as e:
        print('Caught expected exception \n  {}: {}'.format(error_class, e))
    except Exception as e:
        print('Got unexpected exception \n  {}: {}'.format(type(e), e))
    else:
        raise Exception('Expected {} to be raised but no error was raised!'.format(error_class))

@tf.function
def add(a, b):
    return a + b

c = add(tf.ones([2, 2]), tf.ones([2, 2]))
print(c)

v = tf.Variable(1.0)
with tf.GradientTape() as tape:
    result = add(v, 1.0)
c = tape.gradient(result, v)
print(c)


@tf.function
def dense_layer(x, w, b):
    return add(tf.matmul(x, w), b)
c = dense_layer(tf.ones([3, 2]), tf.ones([2, 2]), tf.ones([2]))
print(c)


@tf.function
def double(a):
    print('Tracing with', a)
    return a+a

print(double(tf.constant(1.)))
print()
print(double(tf.constant(1.1)))
print()
print(double(tf.constant('a')))
print()












