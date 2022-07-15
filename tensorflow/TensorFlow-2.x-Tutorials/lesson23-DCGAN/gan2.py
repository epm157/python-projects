import  tensorflow as tf
from    tensorflow import keras


class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        self.n_f = 512
        self.n_k = 4

        self.dense1 = keras.layers.Dense(3 * 3 * self.n_f)
        self.conv2 = keras.layers.Conv2DTranspose(self.n_f // 2, 3, 2, 'valid')
        self.bn2 = keras.layers.BatchNormalization()
        self.conv3= keras.layers.Conv2DTranspose(self.n_f // 4, self.n_k, 2, 'same')
        self.bn3 = keras.layers.BatchNormalization()
        self.conv4 = keras.layers.Conv2DTranspose(1, self.n_k, 2, 'same')

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = tf.reshape(x, shape=[-1, 3, 3, self.n_f])
        x = tf.nn.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.leaky_relu(x)
        x = self.conv4(x)
        x = tf.tanh(x)
        return x

class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.n_f = 64
        self.n_k = 4

        self.conv1 = keras.layers.Conv2D(self.n_f, self.n_k, 2, 'same')
        self.conv2 = keras.layers.Conv2D(self.n_f * 2, self.n_k, 2, 'same')
        self.bn2 = keras.layers.BatchNormalization()
        self.conv3 = keras.layers.Conv2D(self.n_f * 4, self.n_k, 2, 'same')
        self.bn3 = keras.layers.BatchNormalization()
        self.flatten4 = keras.layers.Flatten()
        self.dense4 = keras.layers.Dense(1)

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = tf.nn.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.leaky_relu(x)
        x = self.flatten4(x)
        x = self.dense4(x)
        return x



