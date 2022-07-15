import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV



dataset = pd.read_csv('test_appdata10.csv')
#dataset['dayofweek'] = pd.to_numeric(dataset['dayofweek'])

dataset = dataset.apply(pd.to_numeric)
dataset = dataset.astype(float)

response = dataset['enrolled']
dataset = dataset.drop(columns = 'enrolled')

#dataset = dataset.iloc[:, :].values.astype(float)
#response = response.iloc[:].values.astype(float)



X_train, X_test, y_train, y_test = train_test_split(dataset, response, test_size = 0.2, random_state = 0)


train_idendity = X_train['user']
X_train = X_train.drop(columns = ['user'])
test_idendity = X_test['user']
X_test = X_test.drop(columns = ['user'])

sc_X = StandardScaler()

'''
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test))
X_train2.columns = X_train.columns.values
X_test.columns = X_test.columns.values
X_train2.index = X_train.index.values.astype(float)
X_test2.index = X_test.index.values.astype(float)
X_train = X_train2
X_test = X_test2
'''

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

classifier = LogisticRegression(solver='saga', penalty = 'l1', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))

'''

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
plt.show()
'''

print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))

accuracies = cross_val_score(estimator = classifier, X = X_train,
                             y = y_train, cv = 10)

print("SVM Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))

final_results = pd.concat([y_test, test_idendity], axis = 1).dropna()
final_results['predicted_reach'] = y_pred
final_results = final_results[['user', 'enrolled', 'predicted_reach']].reset_index(drop = True)
print(final_results)

penalty = ['l1', 'l2']
C = [0.001, 0.01, 0.1, 0.5, 0.9, 1, 2, 5, 10, 100, 1000]
parameters = dict(C = C, penalty = penalty)

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)

t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

print(grid_search.best_score_)
print(grid_search.best_params_)
print(grid_search.best_score_)




