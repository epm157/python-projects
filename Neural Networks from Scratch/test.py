import numpy as np

a = np.array([[1,2],[3,4]])
b = np.array([[11,12],[13,14]])
c = np.dot(a,b)



inputs = [[1.2, 1.5, 3.5, -2.1],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.3, 3.4, -0.6]]
inputs = np.array(inputs)

weights4 = [[3.1, 2.5, -0.8, 2.4],
            [2.4, 2.5, -0.3, 3.1],
            [1.4, 3.2, -0.3, 1.4]]
weights = np.transpose(weights4)
bias = [2.3, 4, -1, 2]

outputs = []

t1 = inputs.dot(weights)
t2 = weights.dot(inputs)
t3 = np.matmul(inputs, weights)

outputs = np.dot(weights, inputs)
# for neuron_weights, neuron_bias in zip(weights4, bias):
#     neuron_output = 0
#     for n_input, weight in zip(inputs, neuron_weights):
#         neuron_output += n_input*weight
#     neuron_output += neuron_bias
#     layer_outputs.append(neuron_output)
print(inputs)
print(weights)
print(outputs)