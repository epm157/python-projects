from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import time
import tempfile


print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))
print(tf.square(2)+tf.square(3))



x = tf.matmul([[1]], [[2, 3]])
print(x)
print(x.shape)
print(x.dtype)


ndarray = np.ones([3, 3])
print(ndarray)
tensor = tf.multiply(ndarray, 42)
print(tensor)


print(np.add(tensor, 1))

print(tensor.numpy())


x = tf.random.uniform([3, 3])

print("Is there a GPU available: "),
print(tf.test.is_gpu_available())

print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))


def time_mul(x):
    start = time.time()
    for i in range(10):
        tf.matmul(x, x)
    result = time.time()-start
    print("10 loops: {:0.2f}ms".format(1000 * result))

with tf.device('CPU:0'):
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith('CPU:0')
    time_mul(x)


ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
    f.write('''Line 1
    Line 2
    Line 3''')
ds_file = tf.data.TextLineDataset(filename)

ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)

print('Elements of ds_tensors:')
for x in ds_tensors:
    print(x)


print('\nElements in ds_file:')
for x in ds_file:
    print(x)



layer = tf.keras.layers.Dense(10, input_shape=(None, 5))
print(layer(tf.zeros([10, 5])))


print(layer.variables)

print(layer.kernel)
print(layer.bias)


class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_variable('kernel', shape=[int(input_shape[-1]), self.num_outputs])

    def call(self, input):
        return tf.matmul(input, self.kernel)

layer = MyDenseLayer(10)
print(layer(tf.zeros([10, 5])))
print(layer.trainable_variables)


class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()


    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(input_tensor)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(input_tensor)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)

block = ResnetIdentityBlock(1, [1, 2, 3])
print(block(tf.zeros([1, 2, 3, 3])))
print([x.name for x in block.trainable_variables])









x = tf.ones((2, 2))
y = tf.reduce_sum(x)
print(x)
print(y)

with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)

dz_dx = t.gradient(z, y)
print(dz_dx.numpy())


x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y = x*x
    z = y*y

dz_dx = t.gradient(z, x)
dy_dx = t.gradient(y, x)
del t
print(dz_dx)
print(dy_dx)



def f(x, y):
    output = 1.0
    for i in range(y):
        if i>1 and i<5:
            output = tf.multiply(output, x)
    return output

def grad(x, y):
    with tf.GradientTape() as t:
        t.watch(x)
        out = f(x, y)
    return t.gradient(out, x)

x = tf.convert_to_tensor(2.0)

print(grad(x, 4))




x = tf.Variable(1.0)

with tf.GradientTape() as t1:
    with tf.GradientTape() as t2:
        y = x*x*x
    dy_dx = t2.gradient(y, x)
d2y_dx = t1.gradient(dy_dx, x)

print(dy_dx.numpy())
print(d2y_dx.numpy())



