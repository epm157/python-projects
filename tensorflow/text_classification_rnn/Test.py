from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow_datasets as tfds
import tensorflow as tf

import matplotlib.pyplot as plt

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
train_dataset = dataset['train']
test_dataset = dataset['test']
tokenizer = info.features['text'].encoder
print('Vocabulary size: {}'.format(tokenizer.vocab_size))



sample_string = 'TensorFlow is cool.'
tokenized_string = tokenizer.encode(sample_string)
print('Tokenized string is {}'.format(tokenized_string))
original_string = tokenizer.decode(tokenized_string)
print('The original string: {}'.format(original_string))

assert original_string == sample_string

for ts in tokenized_string:
    print('{} ----> {}'.format(ts, tokenizer.decode([ts])))

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, train_dataset.output_shapes)
test_dataset = test_dataset.padded_batch(BATCH_SIZE, test_dataset.output_shapes)


# for example in tfds.as_numpy(train_dataset):
#     text = example[0]
#     text_length = len(text)
#     text_shape = text.shape
#     text = tokenizer.decode(text)
#     target = example[1]
#     ex = example

for batch in tfds.as_numpy(train_dataset):
    review = batch[0][0]
    target = batch[1][0]
    review_length = len(review)
    review = tokenizer.decode(review)
    print(review)


model = tf.keras.Sequential([tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
                              tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
                              tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
                              tf.keras.layers.Dense(64, activation='relu'),
                              tf.keras.layers.Dense(1, activation='sigmoid')])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)

test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))



def pad_to_zero(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec

def sample_predict(sentence, pad):
    tokenized_sample_pred_text = tokenizer.encode(sentence)

    if pad:
        tokenized_sample_pred_text = pad_to_zero(tokenized_sample_pred_text, 64)

    predictions = model.predict(tf.expand_dims(tokenized_sample_pred_text, 0))

    return predictions

#len(pad_to_zero(tokenizer.encode('Hi my friend'), 25))

sample_pred_text = 'The movie was cool. The animation and the graphics were out of this world. I would recommend this movie.'
predictions_wo_padding = sample_predict(sample_pred_text, False)
predictions_with_padding = sample_predict(sample_pred_text, True)
print(predictions_wo_padding)
print(predictions_with_padding)

sample_pred_text = 'The movie was not good. The animation and the graphics were terrible. I would not recommend this movie.'
predictions_wo_padding = sample_predict(sample_pred_text, False)
predictions_with_padding = sample_predict(sample_pred_text, True)
print(predictions_wo_padding)
print(predictions_with_padding)

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')





























