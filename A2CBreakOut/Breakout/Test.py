import torch
import numpy as np


a = torch.randn((1,5))
print(a)


b = a.pow(2)
print(b)

c = a.sum(1, keepdim=True)
print(c)

d = a.pow(2).sum(1, keepdim=True)
print(d)

e = a.pow(2).sum(1, keepdim=True).expand_as(a)
print(e)

f = a * 0.9 / e
print(f)




a *= 0.9 / torch.sqrt(a.pow(2).sum(1, keepdim=True).expand_as(a))
print(a)




b = a.pow(2).sum(1)
print(b)



import numpy as np
x = np.array([1, 2, 3], dtype = np.uint8)

# applying function
print(np.prod(x))
