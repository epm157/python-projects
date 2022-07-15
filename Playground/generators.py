from time import sleep



myGenerator = (x*2 for x in range(10))

for i in myGenerator:
    print(i)

def createGenerator():
    myList = range(5)
    for i in myList:
        yield  i * 3

for i in createGenerator():
    print(i)







def my_gen():
    n = 0
    n = n + 1
    yield n
    n += 1
    yield n
    n += 1
    yield n

gen = my_gen()
print(next(gen))
print(next(gen))
print(next(gen))

for i in my_gen():
    print(i)

def reverse_string(value):
    length = len(value)
    for i in range(length-1, -1, -1):
        yield value[i]

for char in reverse_string('Ehsan'):
    print(char)


my_list = [1, 3, 6, 10]
my_list_ = [x**2 for x in my_list]
my_list_gen = (x**2 for x in my_list)

#generator = my_list_gen()
print(next(my_list_gen))
print(next(my_list_gen))
print('*' * 10)

for item in my_list_gen:
    print(item)

print('*' * 10)

def fibonacci_numbers(num):
    a, b = 0, 1
    for i in range(num):
        a, b = b, a + b
        yield b
def square(nums):
    for num in nums:
        yield num*2

my_gen = fibonacci_numbers(10)
for i in my_gen:
    sleep(0.1)
    print(i)

print(sum(square(fibonacci_numbers(50))))

