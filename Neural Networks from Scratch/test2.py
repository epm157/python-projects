import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(y.shape)
        self.lr = 0.05

    def feedForward(self):
        self.z1 = np.dot(self.input, self.weights1)
        self.layer1 = sigmoid(self.z1)
        self.z2 = np.dot(self.layer1, self.weights2)
        self.output = sigmoid(self.z2)

    def backProp(self):
        error = self.output - self.y
        error *= 2

        dz2 = sigmoid_der(self.z2)
        delta2 = error * dz2
        inps = self.layer1.T
        x = np.dot(inps, delta2)
        self.weights2 -= self.lr * x

        dl1 = np.dot(delta2, self.weights2.T)
        dz1 = sigmoid_der(self.layer1)

        delta1 = dl1 * dz1
        inps = self.input.T
        x = np.dot(inps, delta1)
        self.weights1 -= self.lr * x
        return error.sum()

    '''
            # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
            d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
            d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

            # update the weights with the derivative (slope) of the loss function
            self.weights1 += d_weights1
            self.weights2 += d_weights2
           '''


feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
labels = np.array([[1,0,0,1,1]])
labels = labels.reshape(5,1)

model = NeuralNetwork(feature_set, labels)

for epoch in range(200_000):

    model.feedForward()
    err = model.backProp()
    if epoch % 100 == 0:
        print(err)