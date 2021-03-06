import tensorflow as tf
import matplotlib.pylab as plt
import numpy as np
import time

from tensorflow.keras.callbacks import TensorBoard


NAME = 'mnist-{}'.format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


plt.imshow(x_train[0])
plt.show()

print(x_train[0])



model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=300, callbacks=[tensorboard])



val_loss, val_aacuracy = model.evaluate(x_test, y_test)
print(val_loss)
print(val_aacuracy)

model.save('num_reader.model')

new_model = tf.keras.models.load_model('num_reader.model')

predictions = model.predict([x_test])
print(predictions)

print(np.argmax(predictions[0]))

plt.imshow(x_test[0])
plt.show()






