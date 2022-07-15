import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

dataset = pd.read_csv('/home/ehsan/Dropbox/junk/ML/Machine Learning Practical 6 Real-World Applications/code/7. Credit Card Fraud Detection/Jupyter notebooks/creditcard.csv')

print(dataset.head(10))

dataset2 = dataset.drop(columns=['Class', 'Time'])




fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(dataset2.shape[1]):
    plt.subplot(6, 5, i + 1)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i])

    vals = np.size(dataset2.iloc[:, i].unique())
    if vals >= 100:
        vals = 100

    plt.hist(dataset2.iloc[:, i], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

dataset2.corrwith(dataset.Class).plot.bar(
    figsize=(20, 10), title='Correlation with E Signed',
    fontsize=15, rot=45, grid=True)
plt.show()

## Correlation Matrix
sn.set(style="white")

# Compute the correlation matrix
corr = dataset2.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))

# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
           square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()





am = dataset['Amount'].values
am2 = dataset.Amount.values
# print(am.shape)
am = am.reshape(-1, 1)
# print(am.shape)

scaler = StandardScaler()

dataset['Amount'] = scaler.fit_transform(dataset.Amount.values.reshape(-1, 1))
# data['scaledAmount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

dataset = dataset.drop(columns=['Time'], axis=1)

X = dataset.iloc[:, dataset.columns != 'Class']
y = dataset.iloc[:, dataset.columns == 'Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)




inputDim = X_train.shape[1]
model = Sequential([
    Dense(units=16, input_dim=inputDim, activation='relu'),
    Dense(units=16, activation='relu'),
    Dropout(0.5),
    Dense(units=20, activation='relu'),
    Dense(units=24, activation='sigmoid'),
    Dense(units=1, activation='relu'),
])

print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=15, epochs=5)

score = model.evaluate(X_test, y_test)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()



y_pred = model.predict(X_test)
y_test = pd.DataFrame(y_test)


cm = confusion_matrix(y_test, y_pred.round())
plot_confusion_matrix(cm, [0, 1])



random_forest = RandomForestClassifier(n_estimators=10)
random_forest.fit(X_train,y_train.ravel())
y_pred = random_forest.predict(X_test)
print(random_forest.score(X_test,y_test))

cm = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(cm, [0, 1])


y_pred = random_forest.predict(X)
plot_confusion_matrix(y, y_pred)
plot_confusion_matrix(cm, [0, 1])






fraud_indices = np.array(dataset[dataset.Class == 1].index)
number_records_fraud = len(fraud_indices)

normal_indices = dataset[dataset.Class == 0].index

random_normal_indices = np.random.choice(normal_indices, number_records_fraud)
random_normal_indices = np.array(random_normal_indices)

under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])



X_resample, y_resample = SMOTE().fit_resample(X, y.values.ravel())

y_resample = pd.DataFrame(y_resample)
X_resample = pd.DataFrame(X_resample)

X_train, X_test, y_train, y_test = train_test_split(X_resample,y_resample,test_size=0.3)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)






