import tensorflow as tf
import matplotlib.pylab as plt
import numpy as np
import csv

print(tf.__version__)

def plot_series(time, series, format='-', start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


time_step = []
sunspots = []

with open('Sunspots.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        time_step.append(int(row[0]))
        sunspots.append(float(row[2]))

series = np.array(sunspots)
time = np.array(time_step)

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()

split_time = 3000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 64
batch_size = 256
shuffle_buffer_size = 1000


train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

print(train_set)

xs, ys = next(iter(train_set))

for i in range(xs.shape[0]):
    print(xs[i].numpy())
    print(ys[i].numpy())
    print()
    print()


def make_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='causal', activation='relu', input_shape=[None, 1]))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
    model.add(tf.keras.layers.Dense(30, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Lambda(lambda x: x * 400.0))
    return model

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast



tf.keras.backend.clear_session()

model = make_model()
optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=['mae'])
history = model.fit(train_set, epochs=500)



forecast = model_forecast(model, series[..., np.newaxis], window_size)

forecast = forecast[..., -1, 0]

results = forecast[split_time - window_size: -1]

plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)
plot_series(time_valid, results)
plt.show()



err = tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
print(err)


mae = history.history['mae']
loss = history.history['loss']

epochs = range(len(loss))


plt.plot(epochs, mae, 'r')
plt.plot(epochs, loss, 'b')
plt.title('MAE and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(['MAE', 'Loss'])
plt.figure()
plt.show()

zoom_level = 200
epochs_zoom = epochs[zoom_level:]
mae_zoom = mae[zoom_level:]
loss_zoom = loss[zoom_level:]


epochs_zoom = epochs[200:]
mae_zoom = mae[200:]
loss_zoom = loss[200:]

plt.plot(epochs_zoom, mae_zoom, 'r')
plt.plot(epochs_zoom, loss_zoom, 'b')
plt.title('MAE and Loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(['MAE', 'Loss'])
plt.figure()
plt.show()




