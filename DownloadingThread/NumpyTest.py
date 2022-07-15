import numpy as np


x = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
t = [1, 2, 3]
a = np.array(t)

print(type(t))
print(type(a))
print(a.shape)

print(a)

a = np.zeros((3, 4))
print(a)

a = np.ones((4,5))
print(a)

a = np.full((2,3), 14)
print(a)

d = np.eye(2)
print(d)

e = np.random.random((3,4))
print(e)

a = np.array(x)

b = a[:2, 1:3]
print(b)

row1 = a[1, :]
row2 = a[1:2, :]

print(row1, row1.shape)  # Prints "[5 6 7 8] (4,)"
print(row2, row2.shape)


a = np.array([[1,2], [3, 4], [5, 6]])
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))
print(a[[0, 1, 2], [0, 1, 0]])

print(a[[1], [1]])
print(np.array([a[1,1]]))

a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

b = np.array([0, 2, 0, 1])
print(a[np.arange(4), b])

a[np.arange(4), b] += 10
print(a)

bool_index = (a>10)

print(bool_index)
print(a[bool_index])
print(a[a>10])



a = np.array([1, 2], dtype = np.float)
print(a.dtype)


x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)

print(y)

for i in range(4):
    y[i, :] = x[i, :] + v


print(y)

print(np.add(x, v))



x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4,1))

print(vv)

print(np.add(x, vv))




N = 10
a = np.random.rand(N,N)
b = np.zeros((N, 1))
a[:, 0] = b.T
c = b

'''
x = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
z = np.array(x)
z.shape
(3, 4)
z.reshape(-1)

z.reshape(-1, 4)
'''