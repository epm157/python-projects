#!/usr/bin/python3
import queue
import threading
import ntpath
import urllib.request as urllib2
from random import randint
from urllib.parse import quote
import concurrent.futures
import time

import os,ssl

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

    print("Finished downloading: %s \n" % url)
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
                time.sleep(1)



def getUrl():

    while True:
        #yield "http://www.marand.ir/wp-content/uploads/2019/03/video_2019-03-09_11-36-49.mp4"
        
        i = randint(0, len(urls)) - 1
        yield urls[i]








urls = ['https://rezvanfarsh.ir/uploads/carpets/17/700-reeds-acrylic-carpet-royalblue-Yashar.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/16/700-reeds-acrylic-carpet-blue-Holiday.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/15/700-reeds-acrylic-carpet-midnightblue-Shahyad.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/37/1200-reeds-acrylic-carpet-cream-AfshanSaltanati.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/18/700-reeds-acrylic-carpet-blue-Nila.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/20/700-reeds-acrylic-carpet-cream-Sogand.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/13/700-reeds-acrylic-carpet-midnightblue-Afshan.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/36/1200-reeds-acrylic-carpet-tan-Respina.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/38/1200-reeds-acrylic-carpet-midnightblue-Holiday.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/19/700-reeds-acrylic-carpet-cream-Mitra.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/12/700-reeds-acrylic-carpet-red-Delsa.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/11/700-reeds-acrylic-carpet-blue-Mahi.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/10/700-reeds-acrylic-carpet-midnightblue-Ahoora.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/14/700-reeds-acrylic-carpet-blue-Kahkeshan.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/43/1200-reeds-acrylic-carpet-tan-BaghMalek.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/41/1200-reeds-acrylic-carpet-blue-BaghMoalaq.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/47/1200-reeds-acrylic-carpet-firebrick-Parinaz.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/45/1200-reeds-acrylic-carpet-midnightblue-Hana.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/46/1200-reeds-acrylic-carpet-cream-Nila.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/44/1200-reeds-acrylic-carpet-gold-Selin.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/9/700-reeds-acrylic-carpet-cream-Negin.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/8/700-reeds-acrylic-carpet-red-Araz.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/91/500-reeds-acrylic-carpet-cream-AfshanGolriz.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/119/Embossed-design-1600-reeds-acrylic-carpet-bazaar-Kheshti.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/90/500-reeds-acrylic-carpet-midnightblue-Soltan.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/89/500-reeds-acrylic-carpet-walnuts-BaghBehesht.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/88/500-reeds-acrylic-carpet-cream-Arshida.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/87/500-reeds-acrylic-carpet-midnightblue-Baharan.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/79/Embossed-design-700-reeds-acrylic-carpet-silver-Sadaf.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/78/Embossed-design-700-reeds-acrylic-carpet-midnightblue-Lirose.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/86/500-reeds-acrylic-carpet-cream-Artin.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/118/Embossed-design-1600-reeds-acrylic-carpet-white-AfshanSaltanati.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/75/Embossed-design-700-reeds-acrylic-carpet-silver-Taha.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/77/Embossed-design-700-reeds-acrylic-carpet-silver-Tabriz.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/76/Embossed-design-700-reeds-acrylic-carpet-midnightblue-Saghi.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/74/Embossed-design-700-reeds-acrylic-carpet-midnightblue-Rozha.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/72/Embossed-design-700-reeds-acrylic-carpet-midnightblue-Bostan.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/73/Embossed-design-700-reeds-acrylic-carpet-silver-Nahal.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/117/Embossed-design-1600-reeds-acrylic-carpet-green-Roza.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/116/Embossed-design-1600-reeds-acrylic-carpet-bazaar-Sarina.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/115/Embossed-design-1600-reeds-acrylic-carpet-cream-Kerman.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/114/Embossed-design-1600-reeds-acrylic-carpet-white-Arghavan.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/113/Embossed-design-1600-reeds-acrylic-carpet-blue-Binazir.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/112/Embossed-design-1600-reeds-acrylic-carpet-cream-Keyhan.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/111/Embossed-design-1600-reeds-acrylic-carpet-bazaar-Golestan.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/85/500-reeds-acrylic-carpet-cream-Afshan2.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/82/500-reeds-acrylic-carpet-midnightblue-Holiday.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/84/500-reeds-acrylic-carpet-blue-Adrina.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/83/500-reeds-acrylic-carpet-cream-Chichak.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/80/500-reeds-acrylic-carpet-cream-Afshan.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/81/500-reeds-acrylic-carpet-midnightblue-BaghMoallagh.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/125/Embossed-design-1500-reeds-acrylic-carpet-bazaar-Tabriz.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/124/Embossed-design-1500-reeds-acrylic-carpet-red-Mastane.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/126/Embossed-design-1500-reeds-acrylic-carpet-cream-Tarannom.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/123/Embossed-design-1500-reeds-acrylic-carpet-cream-Mahro.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/122/Embossed-design-1500-reeds-acrylic-carpet-bazaar-Khatibi.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/121/Embossed-design-1500-reeds-acrylic-carpet-red-GolAfshan.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/120/Embossed-design-1500-reeds-acrylic-carpet-cream-Esfahan.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/67/Embossed-design-1200-reeds-acrylic-carpet-steelblue-Yazdan.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/66/Embossed-design-1200-reeds-acrylic-carpet-silver-KhaneRoya.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/65/Embossed-design-1200-reeds-acrylic-carpet-burlywood-Elsa.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/61/Embossed-design-1200-reeds-acrylic-carpet-cream-AfshanLuxury.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/62/Embossed-design-1200-reeds-acrylic-carpet-bazaar-Delvan.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/59/Embossed-design-1200-reeds-acrylic-carpet-sealbrown-Dayan.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/58/Embossed-design-1200-reeds-acrylic-carpet-silver-Azarmehr.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/60/Embossed-design-1200-reeds-acrylic-carpet-cream-AfshanAfshary.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/68/Embossed-design-1200-reeds-acrylic-carpet-burlywood-Morvarid.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/57/Embossed-design-1200-reeds-acrylic-carpet-steelblue-Pichak.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/64/Embossed-design-1200-reeds-acrylic-carpet-white-Golfam.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/63/Embossed-design-1200-reeds-acrylic-carpet-cream-Sofi.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/40/1200-reeds-acrylic-carpet-cream-Pazel.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/42/1200-reeds-acrylic-carpet-cream-Tabriz.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/39/1200-reeds-acrylic-carpet-cream-Afshan.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/7/700-reeds-acrylic-carpet-cream-Sonia.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/6/700-reeds-acrylic-carpet-blue-Sepideh.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/5/700-reeds-acrylic-carpet-red-Gisoo.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/4/700-reeds-acrylic-carpet-blue-Doryan.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/3/700-reeds-acrylic-carpet-blue-Ghazal.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/2/700-reeds-acrylic-carpet-red-Afsane.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/1/700-reeds-acrylic-carpet-cream-Erfan.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/71/Embossed-design-1200-reeds-acrylic-carpet-steelblue-Vagere.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/70/Embossed-design-1200-reeds-acrylic-carpet-bazaar-Holiday.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/69/Embossed-design-1200-reeds-acrylic-carpet-cream-KimiaSadat.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/55/1200-reeds-acrylic-carpet-cream-KimiaSadat.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/54/1200-reeds-acrylic-carpet-cream-Shamse.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/52/1200-reeds-acrylic-carpet-blue-Kereshmeh.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/53/1200-reeds-acrylic-carpet-midnightblue-QabKheshti.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/51/1200-reeds-acrylic-carpet-midnightblue-Shahcheraq.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/50/1200-reeds-acrylic-carpet-chocolatecosmos-Mehregan.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/48/1200-reeds-acrylic-carpet-tan-HozNoghre.jpg',
         'https://rezvanfarsh.ir/uploads/carpets/49/1200-reeds-acrylic-carpet-cream-Simorgh.jpg']
'''


urls = ["https://host2.rjmusicmedia.com/media/music_video/hd/googoosh-siavash-ghomayshi-40-saal.mp4",
                "https://host2.rjmusicmedia.com/media/podcast/mp3-192/Playout-7.mp3",
                "https://host2.rjmusicmedia.com/media/podcast/mp3-192/Khodcast-2-16-2.mp3",
                "https://host2.rjmusicmedia.com/media/podcast/mp3-192/TranceForm-65.mp3",
                "https://host2.rjmusicmedia.com/media/podcast/mp3-192/Summer-Mix-2018-DJ-Taba-DeeJay-AL.mp3",
                "https://host2.rjmusicmedia.com/media/music_video/4k/aref-be-to-hedyeh-mikonam.mp4",
                "https://host2.rjmusicmedia.com/media/music_video/hd/morvarid-ba-to.mp4",
                "https://host2.rjmusicmedia.com/media/music_video/hd/sahar-che-haliye.mp4",
                "https://host2.rjmusicmedia.com/media/music_video/hd/parsalip-pore-dood-(ft-rudebeny).mp4",
                "https://host2.rjmusicmedia.com/media/music_video/4k/arsalan-aramesh.mp4",
                "https://host2.rjmusicmedia.com/media/music_video/hd/donya-maghroor-(teaser).mp4",
                "https://host2.rjmusicmedia.com/media/music_video/hd/amirabbas-golab-koodakaneh.mp4",
                "https://host2.rjmusicmedia.com/media/music_video/hd/ashvan-moghaser.mp4",
                "https://host2.rjmusicmedia.com/media/music_video/hd/sami-beigi-be-to-marboot-nist.mp4",
                "https://host2.rjmusicmedia.com/media/music_video/hd/hana-eshtebahi-(behind-the-scenes).mp4",
                "https://host2.rjmusicmedia.com/media/music_video/hd/hamid-sefat-man-divane-nistam.mp4"]
'''

if __name__ == "__main__":

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(50) as executor:
        executor.map(download, getUrl())
    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} second(s)\n')


'''

    threads = []

    for i in range(50):
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
    
'''

