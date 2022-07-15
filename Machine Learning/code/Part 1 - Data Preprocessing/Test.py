
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


dataset = pd.read_csv('DataTemp.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values


imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

labelEncoderX = LabelEncoder()
X[:, 0] = labelEncoderX.fit_transform(X[:, 0])

oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray()

labelEncoderY = LabelEncoder()
Y = labelEncoderY.fit_transform(Y)

from sklearn.model_selection import train_test_split

Xtrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
standardScalerX = StandardScaler()
Xtrain = standardScalerX.fit_transform(Xtrain)
XTest = standardScalerX.transform(XTest)
standardScalerY = StandardScaler()
#YTrain = standardScalerY.fit_transform(YTrain)





