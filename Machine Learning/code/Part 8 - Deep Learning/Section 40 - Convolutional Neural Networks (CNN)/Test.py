from keras.layers import Convolution2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator



image_dimension = 64

classifier = Sequential()

classifier.add(Convolution2D(64, 3, 3, input_shape=(image_dimension, image_dimension, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(64, 3, 3, activation='relu'))

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
    epochs = 25,
    validation_data = test_set,
    validation_steps = 2000)
