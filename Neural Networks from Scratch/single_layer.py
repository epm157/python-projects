import numpy as np
feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
labels = np.array([[1,0,0,1,1]])
labels = labels.reshape(5,1)


np.random.seed(42)
#weights = np.random.rand(3,1)
weights = np.random.rand(feature_set.shape[1], 1)
bias = np.random.rand(1)
lr = 0.05


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

for epoch in range(200_000):

    z = np.dot(feature_set, weights) + bias
    pred = sigmoid(z)
    error = pred - labels
    error = 2 * error
    dz = sigmoid_der(z)
    delta = error * dz
    inputs = feature_set.T
    x = np.dot(inputs, delta)

    weights -= lr * x

    for num in delta:
        bias -= lr * num

    if epoch % 10000 == 0:
        print(error.sum())

