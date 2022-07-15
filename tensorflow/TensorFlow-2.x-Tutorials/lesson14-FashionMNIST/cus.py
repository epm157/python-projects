import  os
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, optimizers, datasets

def prepare_mnist_features_and_labels(x, y):
  x = tf.cast(x, tf.float32) / 255.0
  y = tf.cast(y, tf.int64)
  return x, y

def mnist_dataset():
  (x, y), _ = datasets.fashion_mnist.load_data()
  ds = tf.data.Dataset.from_tensor_slices((x, y))
  ds = ds.map(prepare_mnist_features_and_labels)
  ds = ds.take(20000).shuffle(20000).batch(100)
  return ds



@tf.function
def compute_loss(logits, labels):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(loss)
    return loss

def compute_accuracy(logits, labels):
    preditions = tf.argmax(logits, axis=1)
    accuracy = tf.equal(preditions, labels)
    accuracy = tf.cast(accuracy, tf.float32)
    accuracy = tf.reduce_mean(accuracy)
    return accuracy


def train_one_step(model, optimizer, x, y):

    with tf.GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(logits, y)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    accuracy = compute_accuracy(logits, y)

    return loss, accuracy


def train(epoch, model, optimizer):

    train_ds = mnist_dataset()
    loss = 0.0
    accuracy = 0.0
    for step, (x, y) in enumerate(train_ds):
        loss, accuracy = train_one_step(model, optimizer, x, y)
        if step % 500 == 0:
            print('epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())
    return loss, accuracy



class MyModel(keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = layers.Dense(200, activation=tf.nn.relu)
        self.layer2 = layers.Dense(200, activation=tf.nn.relu)
        self.layer3 = layers.Dense(10, activation=tf.nn.relu)

    def call(self, x, training=False):
        x = tf.reshape(x, [-1, 28*28])
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x



def main():
    tf.random.set_seed(22)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


    model = MyModel()

    optimizer = optimizers.Adam()

    for epoch in range(20):
        loss, accuracy = train(epoch, model, optimizer)
    print('Final epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())









if __name__ == '__main__':
    main()
