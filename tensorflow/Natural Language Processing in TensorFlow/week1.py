import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print(tf.__version__)

sentences = ['I love my cat', 'You love my? dog?!', 'Hey man, what do you think about my dog?']

tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)

word_indexes = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')

print(word_indexes)
print(sequences)
print(padded)


test_sentences = ['my cat is my babe']
sequences = tokenizer.texts_to_sequences(test_sentences)
print(sequences)












