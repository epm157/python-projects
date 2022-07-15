import tensorflow as tf

print(tf.__version__)


dataset = tf.data.Dataset.range(10)


for val in dataset:
    print(val.numpy())


dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
for window_dataset in dataset:
    for val in window_dataset:
        print(val.numpy(), end= " ")
    print()

dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(2).prefetch(1)
for x, y in dataset:
    print("x = ", x.numpy())
    print("y = ", y.numpy())




dataset = tf.data.Dataset.range(20)


for value in dataset:
    print(value.numpy())

dataset = dataset.window(10, shift=1)
for window in dataset:
    for value in window:
        print(value.numpy(), end = " ")
    print()



dataset = dataset.window(10, shift=1, drop_remainder=True)
for window in dataset:
    for val in window:
        print(val.numpy(), end=" ")
    print()

dataset = tf.data.Dataset.range(20)
dataset = dataset.window(10, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(10))
for window in dataset:
    print(window.numpy())


dataset = tf.data.Dataset.range(20)
dataset = dataset.window(10, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(10))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(10)
dataset = dataset.batch(2).prefetch(1)
for x, y in dataset:
    print("x = ", x.numpy())
    print("y = ", y.numpy())


















