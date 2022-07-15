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
batch_size = 32
shuffle_buffer_size = 1000


dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
print(dataset)

xs, ys = next(iter(dataset))

for i in range(xs.shape[0]):
    print(xs[i].numpy())
    print(ys[i].numpy())
    print()
    print()





#l0 = tf.keras.layers.Dense(10, input_shape=[window_size])
l0 = tf.keras.layers.Dense(10, input_shape=[window_size], activation='relu')
l1 = tf.keras.layers.Dense(10 , activation='relu')
l2 = tf.keras.layers.Dense(1)
model = tf.keras.models.Sequential([l0, l1, l2])


#model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))


lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch/20))
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)

model.compile(loss='mse', optimizer=optimizer)

#model.fit(dataset, epochs=500, verbose=0)

history = model.fit(dataset, epochs=100)

#print(f'Layer weights {l0.get_weights()}')



lrs = 1e-8 * (10 ** (np.arange(100) / 20))
plt.semilogx(lrs, history.history["loss"])
plt.axis([1e-8, 1e-3, 0, 300])
plt.show()


lrs = 1e-8 * (10 ** np.arange(100) / 20)
plt.semilogx(lrs, history.history['loss'])
plt.axis([1e-8, 1e-3, 0, 300])
plt.show()

loss = history.history['loss']
plot_loss = loss[40:]
epochs = range(40, len(loss))
plt.plot(epochs, plot_loss, 'b', label='Training loss')
plt.show()

loss = history.history['loss']
epochs = range(10, len(loss))
plot_loss = loss[10:]
print(plot_loss)
plt.plot(epochs, plot_loss, 'b', label='Training Loss')
plt.show()




forecast = []

ran = len(series) - window_size
for time in range(ran):
    input = series[time: time+window_size]
    forecast.append(model.predict(input[np.newaxis]))

forecast = forecast[split_time-window_size:]
results = np.array(forecast)
results = results[:, 0, 0]

plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)
plot_series(time_valid, results)

plt.show()

error = tf.keras.metrics.mean_absolute_error(x_valid, results)
print(error.numpy())














