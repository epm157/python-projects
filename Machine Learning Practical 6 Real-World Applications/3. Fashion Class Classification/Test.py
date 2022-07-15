import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Convolution2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.layers import Dropout



fashion_train_df = pd.read_csv('/Users/ehsan/Dropbox/junk/ML/Machine Learning Practical 6 Real-World Applications/3. Fashion Class Classification/fashion-mnist_train.csv', sep = ',')
fashion_test_df = pd.read_csv('/Users/ehsan/Dropbox/junk/ML/Machine Learning Practical 6 Real-World Applications/3. Fashion Class Classification/fashion-mnist_test.csv', sep = ',')

print(fashion_train_df.head())
print(fashion_train_df.shape)



X_train = fashion_train_df.iloc[:, 1:].values.astype(float)
y_train = fashion_train_df.iloc[:, 0].values.astype(float)
X_test = fashion_test_df.iloc[:, 1:].values.astype(float)
y_test = fashion_test_df.iloc[:, 0].values.astype(float)


i = random.randint(1, len(X_train))
plt.imshow(X_train[i, :].reshape((28, 28)), cmap = 'gray')
plt.show()



W_grid = 10
L_grid = 10
fig, axes = plt.subplots(L_grid, W_grid, figsize = (17,17))
axes = axes.ravel() # flaten the 15 x 15 matrix into 225 array
n_training = len(X_train) # get the length of the training dataset
# Select a random number from 0 to n_training
for i in np.arange(0, len(axes)): # create evenly spaces variables (len(axes) = W_grid * L_grid
    # Select a random number
    index = np.random.randint(0, n_training)
    # read and display an image with the selected index
    axes[i].imshow( X_train[index,:].reshape((28,28)), cmap = 'gray')
    axes[i].set_title(y_train[index], fontsize = 8)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)
plt.show()




X_train = X_train/255
X_test = X_test/255


X_train, X_validate, y_train, y_validate = train_test_split(X_train,y_train, test_size = 0.2, random_state = 12345)

image_dimension = 28
X_train = X_train.reshape(X_train.shape[0], *(image_dimension, image_dimension, 1))
X_test = X_test.reshape(X_test.shape[0], *(image_dimension, image_dimension, 1))
X_validate = X_validate.reshape(X_validate.shape[0], *(image_dimension, image_dimension, 1))

cnn_model = Sequential()

cnn_model.add(Convolution2D(128, 3, 3, input_shape=(image_dimension, image_dimension, 1), activation='relu'))

cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

cnn_model.add(Flatten())

cnn_model.add(Dense(units = 64, activation = 'relu'))
cnn_model.add(Dropout(rate=0.1))
cnn_model.add(Dense(units = 64, activation = 'relu'))
cnn_model.add(Dropout(rate=0.1))
cnn_model.add(Dense(units = 10, activation = 'sigmoid'))

cnn_model.compile(loss = 'sparse_categorical_crossentropy', optimizer = Adam(lr = 0.001), metrics = ['accuracy'])

epochs = 5

cnn_model.fit(X_train, y_train, batch_size = 512, nb_epoch = epochs,
              verbose = 1, validation_data = (X_validate, y_validate))


evaluation = cnn_model.evaluate(X_test, y_test)
print('Test Acciracy : {:.3f}'.format(evaluation[1]))

predicted_classes = cnn_model.predict_classes(X_test)

L = 5
W = 5
fig, axes = plt.subplots(L, W, figsize = (12,12))
axes = axes.ravel() #

for i in np.arange(0, L * W):
    axes[i].imshow(X_test[i].reshape(28,28))
    axes[i].set_title("Prediction Class = {:0.1f}\n True Class = {:0.1f}".format(predicted_classes[i], y_test[i]))
    axes[i].axis('off')

plt.subplots_adjust(wspace=0.5)
plt.show()

cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize = (14, 10))
sns.heatmap(cm, annot = True)
plt.show()

num_classes = 10
target_names = ['Class {}'.format(i) for i in range(num_classes)]
print(classification_report(y_test, predicted_classes, target_names = target_names))



