import tensorflow as tf
import matplotlib.pylab as plt
import numpy as np

print(tf.__version__)



def plot_series(time, series, format='-', start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)


def trend(time, slope=0):
    return time * slope

def seasonal_pattern(season_time):
    return np.where(season_time < 0.4, np.cos(season_time * 2 * np.pi), 1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


time = np.arange(4 * 356 + 1, dtype=np.float)

baseline = 10
series = trend(time, 0.1)
amplitude = 40
slope = 0.05
noise_level = 5

series = baseline + trend(time, slope) + seasonality(time, period=356, amplitude=amplitude)
series += noise(time, noise_level, seed=42)

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]


window_size = 20
batch_size = 128
shuffle_buffer_size = 1000


train_dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
print(train_dataset)

xs, ys = next(iter(train_dataset))

for i in range(xs.shape[0]):
    print(xs[i].numpy())
    print(ys[i].numpy())
    print()
    print()


def make_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Lambda(lambda x: x * 100.0))
    return model



model = make_model()
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch/20))
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=['mae'])
history = model.fit(train_dataset, epochs=100, callbacks=[lr_scheduler])

plt.semilogx(history.history['lr'], history.history['loss'])
plt.axis([1e-8, 1e-4, 0, 30])
plt.show()



tf.keras.backend.clear_session()

model = make_model()
optimizer = tf.keras.optimizers.SGD(lr=5e-5, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=['mae'])
history = model.fit(train_dataset, epochs=500)


forecast = []
for time in range(len(series) - window_size):
    input = series[time: time+window_size]
    input = input[np.newaxis]
    pred = model.predict(input)
    forecast.append(pred)

forecast = forecast[split_time-window_size:]
results = np.array(forecast)
results = results[:, 0, 0]


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



