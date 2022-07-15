
import ntpath
import os
import urllib.request  as urllib2
from functools import reduce
from random import randint

def get_file_name(path):
    head, tail = ntpath.split(path)

    return tail or ntpath.basename(head)


def get_file_extension(path):
    filename, file_extension = os.path.splitext(path)
    return file_extension


def download(url):

    fileName = get_file_name(url)
    fileExtension = get_file_extension(url)
    fileNameWOextension = fileName.replace(fileExtension, "")
    i = randint(0, 1000)
    downloadedFileName = '{}_{}{}'.format(fileNameWOextension, i, fileExtension)

    file = urllib2.urlopen(url)

    with open(downloadedFileName, 'wb') as output:
        while True:
            data = file.read(4096)
            if data:
                output.write(data)
            else:
                break
    return downloadedFileName


def delete(filePath):
    try:
        #pass
        os.remove(filePath)
    except OSError as e:  ## if failed, report it back to the user ##
        print("Error: %s - %s." % (e.filename, e.strerror))





l = [1, 5,65, -8, 0]

for i, v in enumerate(l):
    print(i, v)

j = [True, True, False]

print(all(j))
print(any(j))


comp1 = complex(2, 3)

print(comp1)


comp2 = complex("10+1j")

print(comp2)

print(comp1 + comp2)



str = "How long are words in this text"


print(list(map(len, str.split())))

nums = [1, 5, 7, 0]

print(reduce(lambda a, b: a*10 + b, nums))

print(list(filter(lambda s: s[0] == 't', str.split())))

