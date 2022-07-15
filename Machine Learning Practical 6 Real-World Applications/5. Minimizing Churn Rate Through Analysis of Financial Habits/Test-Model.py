import pandas as pd
import numpy as np
import random
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


dataset = pd.read_csv('new_churn_data.csv')

user_identifier = dataset['userid']
dataset = dataset.drop(columns = ['userid'])

print(dataset.rent_or_own.value_counts())

dataset.groupby('rent_or_own')['churn'].nunique().reset_index()

#one_hot_rent_or_own = pd.get_dummies(dataset['rent_or_own'], prefix = 'rent_or_own')
#dataset = dataset.drop('rent_or_own', axis = 1)
#dataset = dataset.join(one_hot_rent_or_own)

dataset = pd.get_dummies(dataset)

dataset = dataset.drop(columns = ['rent_or_own_na', 'zodiac_sign_na', 'payfreq_na'])

X_train, X_test, y_train, y_test = train_test_split(dataset.drop(columns = 'churn'), dataset['churn'],
                                                    test_size = 0.2,
                                                    random_state = 0)
print(y_train.value_counts())

pos_index = y_train[y_train.values == 1].index
neg_index = y_train[y_train.values == 0].index

if len(pos_index) > len(neg_index):
    higher = pos_index
    lower = neg_index
else:
    higher = neg_index
    lower = pos_index

random.seed(0)
higher = np.random.choice(higher, size = len(lower))
lower = np.asarray(lower)
new_indexes = np.concatenate((lower, higher))

X_train = X_train.loc[new_indexes,].astype(np.float64)
y_train = y_train[new_indexes].astype(np.float64)

scaler = StandardScaler()
X_train2 = pd.DataFrame(scaler.fit_transform(X_train))
X_test2 = pd.DataFrame(scaler.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2


classifier = LogisticRegression(solver = 'lbfgs', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
print(accuracy_score(y_pred, y_test))
print(precision_score(y_pred, y_test))
print(recall_score(y_pred, y_test))
print(f1_score(y_pred, y_test))

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
plt.show()
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))


accuracies = cross_val_score(estimator = classifier, X = X_train,
                             y = y_train, cv = 10)
print("SVM Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))


first = pd.DataFrame(X_train.columns, columns = ['features'])
element = classifier.coef_
elementT = np.transpose(element)
second = pd.DataFrame(elementT, columns = ['coef'])
coefs = pd.concat([first, second], axis = 1)

classifier = LogisticRegression(solver = 'lbfgs')
rfe = RFE(estimator=classifier, n_features_to_select=20)
rfe = rfe.fit(X_train, y_train)
print(rfe.support_)
print(rfe.ranking_)
print(X_train.columns[rfe.support_])

# New Correlation Matrix
sn.set(style="white")

# Compute the correlation matrix
corr = X_train[X_train.columns[rfe.support_]].corr()

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


classifier = LogisticRegression(solver = 'lbfgs')
classifier.fit(X_train[X_train.columns[rfe.support_]], y_train)
y_pred = classifier.predict(X_test[X_test.columns[rfe.support_]])


cm = confusion_matrix(y_pred, y_test)
print(accuracy_score(y_pred, y_test))
print(precision_score(y_pred, y_test))
print(recall_score(y_pred, y_test))
print(f1_score(y_pred, y_test))

accuracies = cross_val_score(estimator = classifier,
                             X = X_train[X_train.columns[rfe.support_]],
                             y = y_train, cv = 10)
print("SVM Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))

coefs = pd.concat([pd.DataFrame(X_train[X_train.columns[rfe.support_]].columns, columns=['features']),
                   pd.DataFrame(np.transpose(classifier.coef_), columns=['coef'])],
                  axis = 1)

final_result = pd.concat([y_test, user_identifier],
                         axis=1).dropna()
final_result['predicted_churn'] = y_pred
final_result = final_result[['userid', 'churn', 'predicted_churn']].reset_index(drop=True)


print(X_train[X_train.columns[rfe.support_]].columns)
print(X_train.columns[rfe.support_])
