import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
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
XTest = standardScalerX.transform(XTest)

classifier = Sequential()

classifier.add(Dense(units=6, input_dim=11, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(rate=0.1))

classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(Xtrain, yTrain, batch_size=10, epochs=100)

y_pred = classifier.predict(XTest)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(yTest, y_pred)
print(cm)

"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
customer1 = np.array([[600, 'France', 'Male', 40, 3, 60000, 2, 1, 1, 50000]])
customer1[:, 2] = labelEncoderGender.transform(customer1[:, 2])
customer1 = columnTransformerCountry.transform(customer1)
customer1 = customer1[:, 1:]
customer1 = customer1.astype(float)
customer1 = standardScalerX.transform(customer1)

customer1_pred = classifier.predict(customer1)
print(customer1_pred)

'''
def buildClassifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, input_dim=11, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = buildClassifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator = classifier, X = Xtrain, y = yTrain, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()
'''



def buildClassifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, input_dim=11, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer = optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = buildClassifier)
parameters = {'batch_size': [25, 32],
                'epochs': [100, 200, 500],
                'optimizer': ['adam', 'rmsprop']}

gridSearch = GridSearchCV(estimator = classifier, param_grid = parameters,
                          scoring = 'accuracy', cv = 10, n_jobs = -1)

gridSearch = gridSearch.fit(Xtrain, yTrain)
best_accuracy = gridSearch.best_score_
best_parameters = gridSearch.best_params_
best_estimator = gridSearch.best_estimator_

