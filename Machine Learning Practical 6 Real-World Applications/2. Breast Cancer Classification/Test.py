import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


cancerDataset = load_breast_cancer()
#print (cancerDataset)
print(cancerDataset.keys())
#print(cancerDataset['DESCR'])
print(cancerDataset['target_names'])
print(cancerDataset['target'])
print(cancerDataset['feature_names'])
print(cancerDataset['data'].shape)

df_cancer = pd.DataFrame(np.c_[cancerDataset['data'], cancerDataset['target']], columns = np.append(cancerDataset['feature_names'], ['target']))

print(df_cancer.head())
print(df_cancer.tail())

Example = np.c_[np.array([1,2,3]), np.array([4,5,6])]
print(Example.shape)
print(Example)

a = np.array([[1, 2, 3],
           [4, 5, 6]])
b = np.array([[7, 8, 9],
           [10, 11, 12]])
print(b.shape)
print(np.c_[a, b].shape)

x = np.array([1,2,3])
print(x)
print(x.shape)
sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])
plt.show()

sns.countplot(df_cancer['target'], label = 'Count')
plt.show()

sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)
plt.show()

f = plt.figure()




plt.figure(figsize = (20, 10))
sns.heatmap(df_cancer.corr(), annot = True)
plt.show()

X = df_cancer.drop(['target'], axis = 1)
y = df_cancer['target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 45)

min_train = X_train.min()
range_train = (X_train - min_train).max()

X_train = (X_train - min_train) / range_train

min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test = (X_test - min_test)/range_test

x = X_train['mean area']
y = X_train['mean smoothness']

#sns.scatterplot(x = x, y = y, hue = y_train)
#plt.show()

svc_model = SVC()
svc_model.fit(X_train, y_train)
y_pred = svc_model.predict(X_test)
cm = confusion_matrix(y_pred, y_test)
#sns.heatmap(cm, annot = True, fmt = 'd')
#plt.show()
print(classification_report(y_test, y_pred))

param_grid = {'C' : [1, 10, 100, 1000],
              'gamma' : [10, 5, 3, 2, 1, 0.9, 0.8, 0.5, 0.1, 0.01, 0.001, 0.0001],
              'kernel' : ['rbf']}

grid_search = GridSearchCV(estimator = SVC(), param_grid = param_grid,
                           refit = True, verbose = 4, cv = 10, n_jobs = -1)
grid_search.fit(X_train, y_train)




grid_pred = grid_search.predict(X_test)
cm = confusion_matrix(grid_pred, y_test)

sns.heatmap(cm, annot = True)
plt.show()

print(classification_report(grid_pred, y_test))


print(grid_search.best_score_)
print(grid_search.best_estimator_)
print(grid_search.best_params_)



