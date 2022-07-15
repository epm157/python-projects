
#!/usr/bin/python3


import queue
import threading
import time



class myThread(threading.Thread):

    def __init__(self, name, q):
        threading.Thread.__init__(self)
        self.name = name
        self.q = q

    def run(self):
        print("Starting: " + self.name + "\n")
        while not self.q.empty():
            process_data(self.name, self.q)
        print("Exiting " + self.name + "\n")

def process_data(name, q):
    queueLock.acquire()
    if not workQueue.empty():
        data = q.get()
        queueLock.release()
        print("Processing thread name: " + name + " data: " + data)
        q.task_done()
    else:
        pass
        queueLock.release()
    time.sleep(1)


queueLock = threading.Lock()
workQueue = queue.Queue(15)
threadList = ["Thread-1", "Thread-2"]
nameList = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight"]
threads = []

# Fill the queue
#queueLock.acquire()
for word in nameList:
   workQueue.put(word)
#queueLock.release()

# Create new threads
for tName in threadList:
   thread = myThread(tName, workQueue)
   thread.start()
   threads.append(thread)



while not workQueue.empty():
    pass

#for t in threads:
    #t.join()

workQueue.join()

print ("Exiting Main Thread")


