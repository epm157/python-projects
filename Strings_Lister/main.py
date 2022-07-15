import re



def trim(line):
    start = line.find('"')
    end = line.find('" ')
    result = line[start+1: end]
    return result

def get_names():
    with open('EnglishLocalizable.strings', 'r') as file:
        lines = file.read().splitlines()

    names = [trim(line) for line in lines if len(line) > 0 and not line.startswith('/*') and line.startswith('"')]
    return names

def get_strings():
    with open('EnglishLocalizable.strings', 'r') as file:
        data = file.read().replace("\n", " ")

    matches = re.findall(r".*?=\s\"(.[^\"]*)\";.*?", data)
    return matches





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    names = get_names()
    strings = get_strings()

    MyFile = open('StringsFormat1.txt', 'w')
    for name, string in zip(names, strings):
        MyFile.write(name)
        MyFile.write('\n')
        MyFile.write(string)
        MyFile.write('\n')
        MyFile.write('\n')
        MyFile.write('\n')
    MyFile.close()


