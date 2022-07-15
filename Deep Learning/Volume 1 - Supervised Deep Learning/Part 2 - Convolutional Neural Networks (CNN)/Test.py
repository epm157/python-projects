from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image


image_dimension = 64

classifier = Sequential()

classifier.add(Conv2D(64, (3, 3), input_shape=(image_dimension, image_dimension, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(64, (3, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))

classifier.add(Dense(units=128, activation='relu'))

classifier.add(Dense(units=1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(image_dimension, image_dimension),
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(image_dimension, image_dimension),
    batch_size=32,
    class_mode='binary')

classifier.fit_generator(
    training_set,
    steps_per_epoch = 8000,
    epochs = 5,
    validation_data = test_set,
    validation_steps = 2000)




testImage = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg',
                           target_size = (image_dimension, image_dimension))
testImage = image.img_to_array(testImage)
testImage = np.expand_dims(testImage, axis = 0)
result = classifier.predict(testImage)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
