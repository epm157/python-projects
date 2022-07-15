
from itertools import *

from itsdangerous import izip
from scipy._lib.six import xrange

x = lambda a, b: a*b
print(x(3, 5))


def square_it(a):
    return a*a

x = list(map(square_it, [1, 6, 11]))
print(x)



for i in izip([1, 2, 10], ['p', 'q', 'f']):
    print(i)

print('\n')

for i in izip(count(2), ['Bob', 'Emily', 'Joe']):
    print(i)

print('\n')

def should_drop(x):
    return (x>5)

for i in dropwhile(should_drop, [1, 2, 5, 6, 12]):
    print(i)

print('\n')


a = sorted([1, 2, 1, 3, 2, 1, 2, 3, 4, 5])
for key, value in groupby(a):
    print((key, value), end=' ')

print('\n')

numbers = list()
for i in range(1000000):
    numbers.append(i)

total = sum(numbers)
print(total)
print('\n')

def generate_numbers(n):
    num, numbes = 1, []
    while num < n:
        numbers.append(num)
        num += 1
    return numbers
total = sum(generate_numbers(1000000))

print(total)
print('\n')

total = sum(range(1000+1))
print(total)
print('\n')

total = sum(xrange(1000+1))
print(total)
print('\n')

import numpy as np
import torch
import torchvision
print(torch.__version__)
print(torchvision.__version__)

array = np.arange(12).reshape(3, 4)

m = np.argmax(array, axis=1)



list1 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
list2 = ['a', 'e', 'i']
[print('OK') for e in list1 if e in list2]

print(any((True for x in list1 if x in list2)))

s1 = set(list1)
s2 = set(list2)
result = s1.intersection(s2)
print(result)




a_2d_list = [[1, 2, 3], [4, 5, 6]]
print(a_2d_list[:])
result = [row[0] for row in a_2d_list]
print(result)


array = [1, 2, 1, 3, 2, 1, 2, 3, 4, 5]

def get_first(input):
    return input[0]

result = get_first(array)
print(result)