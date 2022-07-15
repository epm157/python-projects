import urllib.request, urllib.error, urllib.parse
import time

url = 'https://www.mygym.de/covid-19/index'

oldContent = ""

for i in range(100):
    response = urllib.request.urlopen(url, timeout=5)
    webContent = response.read()
    if webContent != oldContent:
        print('Content has changed. New content:')
        print(webContent)
    else :
        print('Content has not changed.')

    oldContent = webContent
    #time.sleep(5)



print('Fiished')
