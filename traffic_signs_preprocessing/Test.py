import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import pandas as pd
import cv2
import timeit


FEATURES_COLUMN = 'features'
LABELS_COLUMN = 'labels'

with open('german-traffic-signs/train.p', 'rb') as f:
    train_data = pickle.load(f)

with open('german-traffic-signs/valid.p', 'rb') as f:
    val_data = pickle.load(f)

with open('german-traffic-signs/test.p', 'rb') as f:
    test_data = pickle.load(f)

X_train = train_data[FEATURES_COLUMN]
y_train = train_data[LABELS_COLUMN]

X_val = val_data[FEATURES_COLUMN]
y_val = val_data[LABELS_COLUMN]

X_test = test_data[FEATURES_COLUMN]
y_test = test_data[LABELS_COLUMN]

print(X_train.shape)
print(X_test.shape)
print(X_val.shape)

assert (X_train.shape[0] == y_train.shape[0]), 'Something went wrong :('
assert(X_train.shape[1:] == (32, 32, 3)),':('

data = pd.read_csv('german-traffic-signs/signnames.csv')

num_of_samples = []

cols = 5
num_classes = 43

def show_grid():
    num_of_samples = []

    cols = 5
    num_classes = 43

    fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 50))
    fig.tight_layout()

    for i in range(cols):
        for j, row in data.iterrows():
            x_selected = X_train[y_train == j]
            axs[j][i].imshow(x_selected[random.randint(0, (len(x_selected) - 1)), :, :], cmap=plt.get_cmap('gray'))
            axs[j][i].axis('off')
            if i == 2:
                axs[j][i].set_title(str(j) + ' - ' + row['SignName'])
                num_of_samples.append(len(x_selected))
    plt.show()
    return num_of_samples


def show_grid2():
    num_of_samples = []

    fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 50))
    fig.tight_layout()

    for j, row in data.iterrows():
        x_selected = X_train[y_train == j]
        for i in range(cols):
            axs[j][i].imshow(x_selected[random.randint(0, (len(x_selected) - 1)), :, :], cmap=plt.get_cmap('gray'))
            axs[j][i].axis('off')
            if i == 2:
                axs[j][i].set_title(str(j) + ' - ' + row['SignName'])
                num_of_samples.append(len(x_selected))

    plt.show()
    return num_of_samples

#num_of_samples = show_grid()
#print(num_of_samples)

num_of_samples = show_grid2()
print(num_of_samples)

#print(timeit.timeit(stmt = show_grid, number = 10))
#print(timeit.timeit(stmt = show_grid2, number = 10))

plt.figure(figsize=(12, 4))
plt.bar(range(num_classes), num_of_samples)
plt.title("Distribution of the train dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()


plt.imshow(X_train[1000])
plt.axis('off')
plt.show()
print(X_train[1000].shape)
print(y_train[1000])

def grayscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

image = grayscale(X_train[1000])
plt.imshow(image)
plt.axis('off')
plt.show()
print(image.shape)


def equilize(image):
    image = cv2.equalizeHist(image)
    return image

image = equilize(image)
plt.imshow(image)
plt.axis('off')
plt.show()
print(image.shape)

def preprocess(image):
    image = grayscale(image)
    image = equilize(image)
    image = image / 255.0
    return image

X_train = np.array(list(map(preprocess, X_train)))




