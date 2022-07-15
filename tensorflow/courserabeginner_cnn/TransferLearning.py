

import tensorflow as tf


print(tf.__version__)
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import datasets, layers, models, Model
import matplotlib.pyplot as plt
import numpy
from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pretrained_model = InceptionV3(input_shape=(300, 300, 3), include_top=False, weights=None)

pretrained_model.load_weights(local_weights_file)

for layer in pretrained_model.layers:
    layer.trainable = False


print(pretrained_model.summary())

last_layer = pretrained_model.get_layer('mixed7')
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pretrained_model.input, x)
model.compile(optimizer=RMSprop(lr=0.0001), loss = 'binary_crossentropy', metrics=['acc'])


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
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(300, 300), batch_size=64, class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(300, 300), batch_size=16, class_mode='binary')

history = model.fit_generator(train_generator, validation_data=validation_generator,
                              steps_per_epoch=16, epochs=25, validation_steps=16, verbose=2)





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