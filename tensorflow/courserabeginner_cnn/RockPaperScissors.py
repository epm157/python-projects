import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import datasets, layers, models, Model
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from os import listdir
from os.path import isfile, join

from tensorflow import keras

print(tf.__version__)

train_dir = '/home/ehsan/Dropbox/junk/ML/tensorflow/beginner/rps/training'
validation_dir = '/home/ehsan/Dropbox/junk/ML/tensorflow/beginner/rps/validation'

train_datagen = ImageDataGenerator(rescale=1/255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale =1 / 255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(300, 300), batch_size=64, class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(300, 300), batch_size=16, class_mode='categorical')


def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    model.compile(optimizer=RMSprop(lr=0.0001), loss='categorical_crossentropy', metrics=['acc'])

    return model

model = create_model()

history = model.fit_generator(train_generator, validation_data=validation_generator,
                                  steps_per_epoch=40, epochs=25, validation_steps=24, verbose=2)

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


a = [1, 2, 3]
x = np.array(a).reshape(1, -1)
y = np.expand_dims(x, axis=0)

p = np.vstack([x])
q = np.vstack(y)

print(x)
print(y)
print(p)
print(q)


model.save_weights('prs_weights')
model.save('prs_model.h5')



def test_model(mdl):
    test_images_dir = '/home/ehsan/Dropbox/junk/ML/tensorflow/beginner/rsp_test_images'
    images_path = [f for f in listdir(test_images_dir) if isfile(join(test_images_dir, f))]
    for path in images_path:
        img = image.load_img(join(test_images_dir, path), target_size=(300, 300))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = mdl.predict(images, batch_size=10)
        print('file name: {}, classified classes: {}\n'.format(path, classes))


model = create_model()
model.load_weights('prs_weights')
#model = keras.models.load_model('prs_model.h5')


test_model(mdl=model)






