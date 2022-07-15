#!/usr/bin/python3
import queue
import threading
import time
import ntpath
import urllib.request as urllib2
from random import randint
from urllib.parse import quote

import os, ssl

#from de.test.aFileUtils import download, delete

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
        getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context





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

    url_trimmed = url.replace(fileNameWOextension, quote(fileNameWOextension))
    file = urllib2.urlopen(url_trimmed)

    while True:
        data = file.read(4096)
        if data:
            # output.write(data)
            pass
        else:
            break
    '''
    
        with open(downloadedFileName, 'wb') as output:
            while True:
                data = file.read(4096)
                if data:
                    #output.write(data)
                    pass
                else:
                    break
        return downloadedFileName
    '''


def delete(filePath):
    try:
        #pass
        os.remove(filePath)
    except OSError as e:  ## if failed, report it back to the user ##
        print("Error: %s - %s." % (e.filename, e.strerror))






class downloadThread(threading.Thread):
    def __init__(self, name, q):
        threading.Thread.__init__(self)
        self.name = name
        self.q = q

    def run(self):
        print("Thread %s started! \n" % self.name)
        while not shouldExit:
            while not self.q.empty():
                url = self.q.get()
                print("%s is downloading: %s \n" % (self.name, url))
                downloadedFileName = download(url)
                #print("Downloaded file name: %s \n" % downloadedFileName)
                #delete(downloadedFileName)
                self.q.task_done()
            else:
                print("Q is empty, %s sleep a bit! \n" % self.name)
                time.sleep(3)



def getUrl():


    urls = ["https://as8.cdn.asset.aparat.com/aparat-video/273d11aaa8de863c56de36b8bc9778bf27812219-1080p.mp4",
                "https://hw13.cdn.asset.aparat.com/aparat-video/08dade035cd829312c6b6159e8a4313726218737-1080p.mp4",
                "https://as6.cdn.asset.aparat.com/aparat-video/ac6950c20dd5c94753b55200af7ae4ed25077945-720p.mp4"]


    i = randint(0, 100)
    return urls[i % len(urls)]


shouldExit = False
threadLock = threading.Lock()
workQueue = queue.Queue(10)

if __name__ == "__main__":

    threads = []

    for i in range(3):
        thread = downloadThread("Thread-{}".format(i), workQueue)
        thread.start()
        threads.append(thread)



    for i in range(2_000_000):
        item = getUrl()
        workQueue.put(item)
        #print("%s is added to the queue, Queue size: %s" % (item, workQueue.qsize()))




    #for t in threads:
        #t.join()

    workQueue.join()

    shouldExit = True
    print("Operation completed!")


