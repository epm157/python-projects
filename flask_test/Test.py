import os
import requests
import numpy as np
import tensorflow as tf

from scipy.misc import imread, imsave
from tensorflow.keras.datasets import fashion_mnist
from flask import Flask, request, jsonify

print(tf.__version__)
'''
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

for i in range(5):
    imsave(f'uploads/{i}.png', arr=X_test[i])
'''

with open('fashion_model_flask.json', 'r') as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)
model.load_weights('fashion_model_flask.h5')

app = Flask(__name__)


@app.route('/api/v1/<string:image_name>', methods=['post'])
def classify_image(image_name):
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    upload_dir = 'uploads/'
    image = imread(upload_dir + image_name)
    image = image.reshape(1, 28*28)
    prediction = model.predict([image])
    prediction = np.argmax(prediction[0])
    result = jsonify({'object_identified': classes[prediction]})
    return result

app.run(port=5000, debug=False)