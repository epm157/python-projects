import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import ntpath
import random
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras import datasets, layers, models


datadir = 'track'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']

data = pd.read_csv(os.path.join('driving_log.csv'), names=columns)
pd.set_option('display.max_colwidth', -1)
print(data.head())

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail

data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)
print(data.head())

num_bins = 25
samples_per_bin = 400
hist, bins = np.histogram(data['steering'], num_bins)
center = (bins[:-1] + bins[1:]) * 0.5
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
plt.show()
print(bins)

print('Total data: ', len(data))
remove_list = []
for j in range(num_bins):
    list_for_bin = []
    for i in range(len(data['steering'])):
        angle = data['steering'][i]
        if angle >= bins[j] and angle <= bins[j+1]:
            list_for_bin.append(i)

    list_for_bin = shuffle(list_for_bin)
    list_for_bin = list_for_bin[samples_per_bin:]
    remove_list.extend(list_for_bin)

print('removed:', len(remove_list))
data.drop(data.index[remove_list], inplace=True)
print('remaining:', len(data))
hist, _ = np.histogram(data['steering'], num_bins)
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
plt.show()

def load_image_steering(datadir):
    images_path = []
    steerings = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
        images_path.append(os.path.join(datadir, center.strip()))
        steerings.append(float(indexed_data[3]))

        images_path.append(os.path.join(datadir, left))
        steerings.append(float(indexed_data[3]) + 0.15)

        images_path.append(os.path.join(datadir, right))
        steerings.append(float(indexed_data[3]) - 0.15)

    images_path = np.asarray(images_path)
    steerings = np.asarray(steerings)

    return images_path, steerings

images_path, steerings = load_image_steering('IMG')
X_train, X_valid, y_train, y_valid = train_test_split(images_path, steerings, test_size=0.2, random_state=6)
print('Training Samples: {}\nValid Samples: {}'.format(len(X_train), len(X_valid)))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y_train, bins=num_bins, width=0.05, color='blue')
axes[0].set_title('Training set')
axes[1].hist(y_valid, bins=bins, width=0.05, color='red')
axes[1].set_title('Validation set')
plt.show()


def image_preprocess(image):
    #image = mpimg.imread(image)
    image = image[60:135, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    image= image / 255.0
    return image


image = images_path[100]
original_image = mpimg.imread(image)
preprocessed_image = image_preprocess(original_image)

fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('Original Image')
axs[1].imshow(preprocessed_image)
axs[1].set_title('Brightness altered image ')
plt.show()


#X_train = np.array(list(map(image_preprocess, X_train)))
#X_valid = np.array(list(map(image_preprocess, X_valid)))

image = mpimg.imread(X_train[random.randint(0, len(X_train) - 1)])
plt.imshow(image)
plt.axis('off')
plt.show()

print(X_train.shape)


'''

def nvidia_model():
    model = Sequential()
    model.add(layers.Conv2D(24, 5, 5, input_shape=(66, 200, 3), activation='relu'))
    model.add(layers.Conv2D(36, 5, 5, activation='relu'))
    model.add(layers.Conv2D(48, 5, 5, activation='relu'))
    model.add(layers.Conv2D(64, 3, 3, activation='relu'))

    model.add(layers.Conv2D(64, 3, 3, activation='relu'))
    #   model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(100, activation='relu'))
    #   model.add(Dropout(0.5))

    model.add(Dense(50, activation='relu'))
    #   model.add(Dropout(0.5))

    model.add(Dense(10, activation='relu'))
    #   model.add(Dropout(0.5))

    model.add(Dense(1))

    optimizer = Adam(lr=1e-3)
    model.compile(loss='mse', optimizer=optimizer)
    return model


model = nvidia_model()
print(model.summary())



'''



def zoom(image):
    zoom = iaa.Affine(scale=(1, 1.3))
    image = zoom.augment_image(image)
    return image

ran = random.randint(0, len(X_train))
image = images_path[ran]
original_image = mpimg.imread(image)
zoomed_image = zoom(original_image)

fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('Original Image')
axs[1].imshow(zoomed_image)
axs[1].set_title('Zoomed image ')
plt.show()


def pan(image):
    pan = iaa.Affine(translate_percent={
        'x': (-0.1, 0.1),
        'y': (-0.1, 0.1)
    })
    image = pan.augment_image(image)
    return image

ran = random.randint(0, len(X_train))
image = images_path[ran]
original_image = mpimg.imread(image)
panned_image = pan(original_image)

fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('Original Image')
axs[1].imshow(panned_image)
axs[1].set_title('Panned image ')
plt.show()


def img_random_brightness(image):
    brightness = iaa.Multiply((0.2, 1.2))
    image = brightness.augment_image(image)
    return image

ran = random.randint(0, len(X_train))
image = images_path[ran]
original_image = mpimg.imread(image)
brightness_altered_image = img_random_brightness(original_image)

fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('Original Image')
axs[1].imshow(brightness_altered_image)
axs[1].set_title('Brightness altered image')
plt.show()

def img_random_flip(image, steering_angle):
    image = cv2.flip(image, 1)
    steering_angle = -steering_angle
    return image, steering_angle

ran = random.randint(0, len(X_train))
image = images_path[ran]
original_image = mpimg.imread(image)
steering_angle = steerings[ran]
flipped_image, flipped_steering_angle = img_random_flip(original_image, steering_angle)

fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('Original Image - ' + 'Steering Angle:' + str(steering_angle))
axs[1].imshow(flipped_image)
axs[1].set_title('Flipped Image - ' + 'Steering Angle:' + str(flipped_steering_angle))
plt.show()

def random_augument(image, steering_angle):
    image = mpimg.imread(image)
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = img_random_brightness(image)
    if np.random.rand() < 0.5:
        image, steering_angle = img_random_flip(image, steering_angle)
    return image, steering_angle

ncol = 2
nrow = 10
fig, axs = plt.subplots(nrow, ncol, figsize=(15, 50))
fig.tight_layout()
for i in range(10):
    ran = random.randint(0, len(X_train))
    image = images_path[ran]
    original_image = mpimg.imread(image)
    steering_angle = steerings[ran]
    augmented_image, flipped_steering_angle = random_augument(image, steering_angle)

    axs[i][0].imshow(original_image)
    axs[i][0].set_title("Original Image")
    axs[i][1].imshow(augmented_image)
    axs[i][1].set_title("Augmented Image")
plt.show()


def batch_generator(images_path, steering_ang, batch_size, is_training):
    while True:
        batch_image = []
        batch_steering = []

        for i in range(batch_size):
            random_index = random.randint(0, len(images_path) - 1)

            if is_training:
                image, steering = random_augument(images_path[random_index], steering_ang[random_index])
            else:
                image = mpimg.imread(images_path[random_index])
                steering = steering_angle[random_index]

            image = image_preprocess(image)
            batch_image.append(image)
            batch_steering.append(steering)
        yield (np.asarray(batch_image), np.asarray(batch_steering))

x_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 1, True))
x_valid_gen, y_test_gen = next(batch_generator(X_valid, y_valid, 1, True))

fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()
axs[0].imshow(x_train_gen[0])
axs[0].set_title('Training Image')

axs[1].imshow(x_valid_gen[0])
axs[1].set_title('Validation Image')
plt.show()














