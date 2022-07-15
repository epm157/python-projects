def foo(*args):
    for arg in args:
        print(arg)

def bar(**kwargs):
    for a in kwargs:
        print(a)
        print(kwargs[a])


l = [['a', 'b'], ['c', 'd', 'e'], ['f', 'g', 'h', 'i']]
mapping = {
    'first': 0,
    'second': 1,
    'third': 2
}
print(l)
print(*l)
foo([1,2], 4)

bar(a=0, b='b')
