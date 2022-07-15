from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf


print(tf.__version__)
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy

flag = False
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if flag:
            print('\nEarly end')
            self.model.stop_training = True

train_dir = '/home/id-th-0755/Dropbox/junk/ML/tensorflow/beginner/images/training'
validation_dir = '/home/id-th-0755/Dropbox/junk/ML/tensorflow/beginner/images/validation'

train_datagen = ImageDataGenerator(rescale=1/255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale =1 / 255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(300, 300), batch_size=128, class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(300, 300), batch_size=32, class_mode='binary')

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())

callback = MyCallback()

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
history = model.fit_generator(train_generator, steps_per_epoch=8, epochs=100, validation_data=validation_generator, validation_steps=8, verbose=2, callbacks=[callback])





plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc='upper right')
plt.show()






