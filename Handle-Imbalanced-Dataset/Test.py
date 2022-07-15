import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
from imblearn.under_sampling import NearMiss
from collections import Counter
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler


rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]


data = pd.read_csv('creditcard.csv',sep=',')
print(data.head())
print(data.head())

target = 'Class'

columns = data.columns.tolist()
columns = [c for c in columns if c not in [target]]

state = np.random.RandomState(42)
X = data[columns]
Y = data[target]
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
print(X.shape)
print(Y.shape)

print(data.isnull().values.any())

count_classes = pd.value_counts(data[target], sort=True)
count_classes.plot(kind='bar', rot=0)
plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()


fraud = data[data[target]==1]
normal = data[data[target]==0]
print(fraud.shape,normal.shape)


nm = NearMiss(random_state=42)
X_res, y_res = nm.fit_resample(X, Y)
print(X_res.shape, y_res.shape)

print('Original dataset shape {}'.format(Counter(Y)))
print('Resampled dataset shape {}'.format(Counter(y_res)))







#Over Sampling 1

data = pd.read_csv('creditcard.csv',sep=',')
print(data.head())
print(data.head())


smk = SMOTETomek(random_state=42)
X_res, y_res = smk.fit_resample(X, Y)

print(X_res.shape, y_res.shape)

print('Original dataset shape {}'.format(Counter(Y)))
print('Resampled dataset shape {}'.format(Counter(y_res)))




#Over Sampling 1

data = pd.read_csv('creditcard.csv',sep=',')
print(data.head())
print(data.head())


os = RandomOverSampler(ratio=0.5)
X_res, y_res = os.fit_resample(X, Y)

print(X_res.shape, y_res.shape)

print('Original dataset shape {}'.format(Counter(Y)))
print('Resampled dataset shape {}'.format(Counter(y_res)))



