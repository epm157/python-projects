import numpy as np
import time


import math

num = 1000000
a = np.random.rand(num)
b = np.random.rand(num)

c = 0
tic = time.time()
for i in range(num):
    c += a[i] * b[i]
toc = time.time()
print(c)
print("Time took: ", (toc - tic) * 1000)


c = 0
tic = time.time()

c = np.dot(a, b)

toc = time.time()
print(c)
print("Time took: ", (toc - tic) * 1000)


def sigmoid(x):
    return 1/(1+math.exp(-x))

sigmoind_v = np.vectorize(sigmoid)

scores = np.array([ -0.54761371,  17.04850603,   4.86054302])
print(sigmoind_v(scores))



a = np.array([56.0 ,0.0, 4.4, 68.0, 1.2, 104.0, 52.0, 8.0, 1.8, 135.0, 99.0, 0.9])
a = np.reshape(a, (3, 4))

sumA = np.sum(a, axis=0)

percentage = a/sumA


a = np.random.randn(10,1)
a = a.reshape((2, 5))
assert(a.shape == (10, 1))