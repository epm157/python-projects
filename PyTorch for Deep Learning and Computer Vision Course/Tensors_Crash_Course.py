import torch
import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)


v = torch.tensor([1, 2, 3, 4, 5, 6])
print(v)
print(v.dtype)

print(v[1:5])
print(v[:-1])

f = torch.FloatTensor([1, 2, 3, 4, 5, 6])
print(f)
print(f.dtype)
print(f.size())

print(f.view(6, -1))
f = f.view(6, -1)
print(f)

a = np.array([5,12,16,145,98])

tensor_cnv = torch.from_numpy(a)
print(tensor_cnv)
print(tensor_cnv.type())

numpy_cnv = tensor_cnv.numpy()
print(numpy_cnv)


t_one = torch.tensor([1, 2, 3])
t_two = torch.tensor([1, 2, 3])
print(t_one + t_two)
print(t_one * t_two ** 2)

print(torch.dot(t_one, t_two))


x = torch.linspace(0, 10, 1000)
#y = torch.exp(x)
#y = x ** 5
y = torch.sin(x)

plt.plot(x.numpy(), y.numpy())
plt.show()



one_d = torch.arange(0, 9)
print(one_d)

two_d = one_d.view(3, -1)
print(two_d)
print(two_d.dim())

print(two_d[0, 0])

print(two_d[1, 2])

print(two_d[0])

print(two_d[:, 0])

x = torch.arange(18).view(3, 2, 3)
print(x)

print(x[1, 1, 1])

print(x[1])

print(x[1, :])

print(x[1, 0:2])


mat_a = torch.tensor([0, 3, 5, 5, 5, 2]).view(2, 3)
mat_b = torch.tensor([3, 4, 3, -2, 4, -2]).view(3, 2)

print(torch.matmul(mat_a, mat_b))


A = np.array([0, 3, 5, 5, 5, 2])
B = np.array([3, 4, 3, -2, 4, -2])
C = np.dot(A, B)
D = np.matmul(A, B)

A = A.reshape(2, 3)
B = B.reshape(3, 2)
C = np.dot(A, B)
D = np.matmul(A, B)



x = torch.tensor(2.0, requires_grad=True)
y = 9*x**4 + 2*x**3 + 3*x**2 + 6*x + 1

y.backward()
print(x.grad)


x = torch.tensor(1.0, requires_grad=True)
z = torch.tensor(2.0, requires_grad=True)

y = x**2 + z**3

y.backward()
print(x.grad)
print(z.grad)















