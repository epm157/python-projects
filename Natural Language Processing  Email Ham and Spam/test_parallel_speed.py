import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import spacy
import re
from tqdm import tqdm
import concurrent.futures
import time


df = pd.read_csv('spam.csv', encoding='latin-1')
textList = df.v2.tolist()
textList = np.array(textList*1)
print(textList.shape)

nlp = spacy.load('en_core_web_sm')

def process_message(msg):
    msg = re.sub('[^a-zA-Z]', ' ', msg)
    msg = msg.lower()
    doc = nlp(msg)
    tokens_no_stop = [token.lemma_ for token in doc if not token.is_stop and not token.text.isspace()]
    msg = ' '.join(tokens_no_stop)
    corpus.append(msg)

def sequentially():
    start = time.perf_counter()
    for message in tqdm(textList):
        message = process_message(message)
        corpus.append(message)
    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} second(s)\n')

def multiThreading():
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_message, textList), total=len(textList)))
    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} second(s)\n')

def multiProcessing():
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_message, textList), total=len(textList)))
    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} second(s)\n')


if __name__ == '__main__':
    corpus = []
    sequentially()
    corpus = []
    multiThreading()
    corpus = []
    multiProcessing()










