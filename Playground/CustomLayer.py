import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression(tf.keras.Model):
    def __init__(self, num_inputs, num_outputs):
        super(LinearRegression, self).__init__()
        self.W = tf.Variable(tf.random_normal_initializer()((num_inputs, num_outputs)))
        self.b = tf.Variable(tf.zeros(num_outputs))
        self.params = [self.W, self.b]

    def call(self, inputs):
        result = tf.matmul(inputs, self.W) + self.b
        return result


N = 100
D = 1
K = 1
X = np.random.random((N, D)) * 2 - 1
w = np.random.randn(D, K)
b = np.random.randn()
Y = X.dot(w) + b + np.random.randn(N, 1) * 0.1

plt.scatter(X, Y)
plt.show()

X = X.astype(np.float32)
Y = Y.astype(np.float32)

def get_loss(model, inputs, targets):
    predictions = model(inputs)
    error = targets - predictions
    result = tf.square(error)
    result = tf.reduce_mean(result)
    return result

def get_grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss = get_loss(model, inputs, targets)
    grad = tape.gradient(loss, model.params)
    return grad


model = LinearRegression(D, K)

print("Initial parameters:")
print(model.W)
print(model.b)


losses = []
optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)

for i in range(100):
    grads = get_grad(model, X, Y)
    optimizer.apply_gradients(zip(grads, model.params))
    loss = get_loss(model, X, Y)
    losses.append(loss)

plt.plot(losses)
plt.show()

x_axis = np.linspace(X.min(), X.max(), 100)
y_axis = model.predict(x_axis.reshape(-1, 1)).flatten()
plt.scatter(X, Y)
plt.plot(x_axis, y_axis)
plt.show()

print("Predicted params:")
print(model.W)
print(model.b)

print("True params:")
print(w)
print(d)