import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

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


classifier = XGBClassifier()
classifier.fit(Xtrain, yTrain)

y_pred = classifier.predict(XTest)

cm = confusion_matrix(yTest, y_pred)
print(cm)

accuracies = cross_val_score(estimator = classifier, X = Xtrain, y = yTrain, cv = 10)
accuracies.mean()
accuracies.std()
