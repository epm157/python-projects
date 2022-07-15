import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

print(tf.__version__)


with open('sarcasm.json', 'r') as f:
    dataset = json.load(f)


sentences = []
labels = []
urls = []
for item in dataset:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])



tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)

word_indexe = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)

print(len(word_indexe))
print(word_indexe)

padded = pad_sequences(sequences, padding='post')

print(sentences[2])
print((padded[2]))
print(labels[2])
print(padded.shape)





