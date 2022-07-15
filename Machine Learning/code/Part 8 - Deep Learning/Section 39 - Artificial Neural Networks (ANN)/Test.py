import pandas as pd
from keras.layers import Dense
import tensorflow
from keras.models import Sequential
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

labelEncoderGender = LabelEncoder()
X[:, 2] = labelEncoderGender.fit_transform(X[:, 2])

columnTransformerCountry = make_column_transformer(
    (OneHotEncoder(categories='auto'), [1]),
    remainder='passthrough')

X = columnTransformerCountry.fit_transform(X)

X = X[:, 1:]

X = X.astype(float)

Xtrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=0)

standardScalerX = StandardScaler()
Xtrain = standardScalerX.fit_transform(Xtrain)
XTest = standardScalerX.fit_transform(XTest)

classifier = Sequential()

classifier.add(Dense(units=6, input_dim=11, kernel_initializer='uniform', activation='relu'))

classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(Xtrain, yTrain, batch_size=10, epochs=100)

y_pred = classifier.predict(XTest)
y_pred = (y_pred > 0.5)

'''
yTest = (yTest > 0)

import numpy as np
cvb = np.array(yTest)
cvb = cvb.reshape((-1, 1))

y_pred = np.array(y_pred)
'''

cm = confusion_matrix(yTest, y_pred)
print(cm)
