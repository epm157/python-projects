
def safe_devide(func):
    def inner(a, b):
        if b == 0:
            print('Devisor should not be zero')
            return
        return func(a, b)
    return inner

@safe_devide
def devide(a, b):
    print(a/b)

def greetings(lang):
    def inner(name):
        if lang == "de":
            return "Hallo " + name
        return "Hi " + name
    return inner


def greeting_factory(lang):
    def decorator(func):
        def inner(name):
            if lang == 'de':
                return func(f'Hallo {name}')
            elif lang == 'en':
                return func(f'Hi {name}')
            else:
                return func(f'Salam {name}')
        return inner
    return decorator


@greeting_factory("de")
def greeting_german(name):
    print(name)

@greeting_factory("en")
def greeting_english(name):
    return name

@greeting_factory("un")
def greeting_unknown(name):
    return name

def divide_wrapper(num):
    if num == 0:
        print('Devisor should not be zero')
        return
    return devide(2, 3)


if __name__ == '__main__':
    devide(4, 0)
    greet_de = greetings("de")
    print(greet_de('Hannah'))
    greeting_german('Mahin')
    print(greeting_english('Ali'))
    print(greeting_unknown('Ehsan'))

    divide_wrapper(0)

