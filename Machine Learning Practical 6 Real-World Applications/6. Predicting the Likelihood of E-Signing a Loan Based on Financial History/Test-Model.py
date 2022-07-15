import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import random
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


random.seed(100)

### Data Preprocessing ###

dataset = pd.read_csv('financial_data.csv')

dataset = dataset.drop(columns=['months_employed'])
dataset['personal_account_months'] = dataset.personal_account_m + (dataset.personal_account_y * 12)
print(dataset[['personal_account_m', 'personal_account_y', 'personal_account_months']].head())
dataset = dataset.drop(columns = ['personal_account_m', 'personal_account_y'])

dataset = pd.get_dummies(dataset)
print(dataset.columns)
dataset = dataset.drop(columns=['pay_schedule_semi-monthly'])

response = dataset['e_signed']
users = dataset['entry_id']
dataset = dataset.drop(columns = ["e_signed", "entry_id"])

X_train, X_test, y_train, y_test = train_test_split(dataset, response,
                                                    test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train2 = pd.DataFrame(scaler.fit_transform(X_train.astype(float)))
X_test2 = pd.DataFrame(scaler.transform(X_test.astype(float)))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2

classifier = LogisticRegression(random_state=0, penalty='l1', solver='saga')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

results = pd.DataFrame([['LinearRegression (Lasso)', acc, prec, rec, f1]],
                       columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_result = pd.DataFrame([['SVM (Linear)', acc, prec, rec, f1]],
                            columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_result, ignore_index=True)

classifier = RandomForestClassifier(random_state=0, n_estimators=100,
                                    criterion='entropy')
classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Random Forest (n=100)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)

accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv = 10, n_jobs=-1)

print("Random Forest Classifier Accuracy: %0.2f (+/- %0.2f)"  % (accuracies.mean(), accuracies.std() * 2))

parameters = {"max_depth": [3, None],
              "max_features": [1, 3, 5, 7, 10],
              'min_samples_split': [2, 5, 8, 10, 12],
              'min_samples_leaf': [1, 2, 3, 5, 10],
              "bootstrap": [True, False],
              "criterion": ['entropy', 'gini']}

grid_search = GridSearchCV(estimator=classifier, param_grid = parameters,
                           scoring='accuracy', cv = 10, n_jobs=-1)

t0 = time.time()

grid_search.fit(X_train, y_train)

t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))


y_pred = grid_search.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Grid Search', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)

final_result = pd.concat([y_test, users], axis = 1).dropna()
final_result['predictions'] = y_pred
final_result = final_result[['entry_id', 'e_signed', 'predictions']]

