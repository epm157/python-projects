from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

np.set_printoptions(precision=3, suppress=True)



with open(train_file_path, 'r') as f:
    names_row = f.readline()

CSV_COLUMNS = names_row.rstrip('\n').split(',')

print(CSV_COLUMNS)


CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']


LABELS = [0, 1]
LABEL_COLUMN = 'survived'


FEATURE_COLUMNS = [column for column in CSV_COLUMNS if column != LABEL_COLUMN]


def get_dataset(file_path):
    dataet = tf.data.experimental.make_csv_dataset(file_path, batch_size=12, label_name=LABEL_COLUMN, na_value='?',
                                                   num_epochs=1, ignore_errors=True)
    return dataet

raw_train_dataset = get_dataset(train_file_path)
raw_test_dataset = get_dataset(test_file_path)

examples, labels = next(iter(raw_train_dataset))
print("EXAMPLES: \n", examples, "\n")
print("LABELS: \n", labels)


CATAGORIES = {
    'sex': ['male', 'female'],
    'class' : ['First', 'Second', 'Third'],
    'deck' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town' : ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone' : ['y', 'n']
}


def process_categorical_data(data, categories):
    data = tf.strings.regex_replace(data, '^ ', '')
    data = tf.strings.regex_replace(data, r'\.$', '')

    data = tf.reshape(data, [-1, 1])
    data = tf.equal(categories, data)
    data = tf.cast(data, tf.float32)
    return data

class_tenosr = examples['class']
print(class_tenosr)

class_categories = CATAGORIES['class']
print(class_categories)

processed_class = process_categorical_data(class_tenosr, class_categories)
print(processed_class)



print("Size of batch: ", len(class_tenosr.numpy()))
print("Number of category labels: ", len(class_categories))
print("Shape of one-hot encoded tensor: ", processed_class.shape)



def process_continous_data(data, mean):
    data = tf.cast(data, tf.float32) * 1/(2*mean)
    return tf.reshape(data, [-1, 1])


MEANS = {
    'age' : 29.631308,
    'n_siblings_spouses' : 0.545455,
    'parch' : 0.379585,
    'fare' : 34.385399
}

age_tensor = examples['age']
print(age_tensor)

print(process_continous_data(age_tensor, MEANS['age']))




def preprocess(features, labels):

    for feature in CATAGORIES.keys():
        features[feature] = process_categorical_data(features[feature], CATAGORIES[feature])

    for feature in MEANS.keys():
        features[feature] = process_continous_data(features[feature], MEANS[feature])

    features = tf.concat([features[column] for column in FEATURE_COLUMNS], 1)

    return features, labels

train_data = raw_train_dataset.map(preprocess).shuffle(500)
test_data = raw_test_dataset.map(preprocess)

examples, labels = next(iter(train_data))
print("EXAMPLES: \n", examples, "\n")
print("LABELS: \n", labels)



def get_model(input_dim, hidden_units=[100, 10]):

    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for unit in hidden_units:
        x = tf.keras.layers.Dense(unit, activation='relu')(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    return model


input_shape, output_shape = train_data.output_shapes
input_dimensions = input_shape.dims[1]

model = get_model(input_dimensions)
print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_data, epochs=20)


test_loss, test_accuracy = model.evaluate(test_data)
print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))


predictions = model.predict(test_data)

for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
    print("Predicted survival: {:.2%}".format(prediction[0]),
          " | Actual outcome: ",
          ("SURVIVED" if bool(survived) else "DIED"))
























