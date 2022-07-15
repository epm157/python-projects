from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np

print(tf.executing_eagerly())


x = [[2.]]
m = tf.matmul(x, x)
print('Hello, {}: '.format(m,))


a = tf.constant([[1, 2], [3, 4]])
print(a)

b = tf.add(a, 1)
print(b)

print(a*b)

c = np.multiply(a, b)
print(c)


def fizzbuzz(max_num):
    counter = tf.constant(0)
    max_num = tf.convert_to_tensor(max_num)
    for num in range(1, max_num.numpy()+1):
        num = tf.constant(num)
        if int(num%3) == 0 and int(num%5) == 0:
            print('FizzBuzz')
        elif int(num%3) == 0:
            print('Fizz')
        elif int(num%5) == 0:
            print('Buzz')
        else:
            print(num.numpy())
        counter +=1

fizzbuzz(15)



class MySimpleLayer(tf.keras.layers.Layer):
    def __init__(self, output_units):
        super(MySimpleLayer, self).__init__()
        self.output_units = output_units
        self.dynamic = True

    def build(self, input_shape):
        self.kernel = self.add_variable('kernal', [input_shape[-1], self.output_units])

    def call(self, input):
        return tf.matmul(input, self.kernel)

model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, input_shape=(784,)),  # must declare input shape
  tf.keras.layers.Dense(10)
])

class MNISTModel(tf.keras.Model):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units=10)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, input):
        result = self.dense1(input)
        result = self.dense2(result)
        result = self.dense2(result)
        return result

model = MNISTModel()






w = tf.Variable([[1.0]])
with tf.GradientTape() as tape:
    loss = w*w
grad = tape.gradient(loss, w)
print(grad)


(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

dataset = tf.data.Dataset.from_tensor_slices((tf.cast(mnist_images[..., tf.newaxis]/255, tf.float32),tf.cast(mnist_labels, tf.int64)))
dataset = dataset.shuffle(1000).batch(32)


mnist_model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(16,[3,3], activation='relu',
                         input_shape=(None, None, 1)),
  tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(10)
])


for images, labels in dataset.take(1):
    print('Logits: ', mnist_model(images[0:1]).numpy())

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_history = []

for(batch, (images, labels)) in enumerate(dataset.take(400)):
    if batch%10 == 0:
        print('.', end='')
    with tf.GradientTape() as tape:
        logits = mnist_model(images, training=True)
        loss_value = loss_object(labels, logits)

    loss_history.append(loss_value.numpy().mean())
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))

import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
plt.show()




class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.W = tf.Variable(5., name='weight')
        self.B = tf.Variable(10., name='bias')

    def call(self, inputs):
        return inputs*self.W+self.B

NUM_EXAMPLES = 2000
training_inputs = tf.random.normal([NUM_EXAMPLES])
noise = tf.random.normal([NUM_EXAMPLES])
training_outputs = training_inputs*3+2+noise

def loss(model, inputs, targets):
    error = model(inputs)-targets
    return tf.reduce_mean(tf.square(error))

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, [model.W, model.B])

model = Model()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))

for i in range(300):
    grads = grad(model, training_inputs, training_outputs)
    optimizer.apply_gradients(zip(grads, [model.W, model.B]))
    if i%20==0:
        print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))

print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))



if tf.test.is_gpu_available():
    with tf.device('gpu:0'):
        v = tf.variable(tf.random.normal([1000, 1000]))
        v = None


x = tf.Variable(10.)
checkpoint = tf.train.Checkpoint(x=x)

x.assign(2.)
checkpoint_path='./ckpt/'
checkpoint.save('./ckpt/')

x.assign(11.0)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))
print(x)


import os

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(10)
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
checkpoint_dir = 'path/to/model_dir'
if not os.path.exists(checkpoint_dir):
  os.makedirs(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
root = tf.train.Checkpoint(optimizer=optimizer,
                           model=model)
root.save(checkpoint_prefix)
root.restore(tf.train.latest_checkpoint(checkpoint_dir))



m = tf.keras.metrics.Mean('loss')
m(0)
m(5)
print(m.result())
m([8, 9])
print(m.result())


def line_search_step(fn, init_x, rate=1.0):
    with tf.GradientTape() as tape:
        tape.watch(init_x)
        value = fn(init_x)
    grad = tape.gradient(value, init_x)
    grad_norm = tf.reduce_mean(grad*grad)
    iniit_value = value
    while value > iniit_value-rate*grad_norm:
        x = init_x-rate*grad
        value=fn(x)
        rate /= 2.0
    return x, value




@tf.custom_gradient
def clip_gradient_bby_norm(x, norm):
    y = tf.identity(x)
    def grad_fn(default):
        return [tf.clip_by_norm(default, norm), None]
    return y, grad_fn


def log1pexp(x):
    return tf.math.log(1+tf.exp(x))

def grad_log1pexp(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        value = log1pexp(x)
    return tape.gradient(value, x)

print(grad_log1pexp(tf.constant(0.)).numpy())

print(grad_log1pexp(tf.constant(100.)).numpy())

@tf.custom_gradient
def log1pexp(x):
    e = tf.exp(x)
    def grad(dy):
        return dy * (1-1/(1+e))
    return tf.math.log(1+e), grad

def grad_log1pexp(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        value = log1pexp(x)
    return tape.gradient(value, x)

print(grad_log1pexp(tf.constant(100.)).numpy())






