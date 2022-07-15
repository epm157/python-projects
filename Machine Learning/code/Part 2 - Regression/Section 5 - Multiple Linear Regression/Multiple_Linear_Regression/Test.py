from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import make_column_transformer

import researchpy as rp




df = pd.read_csv("https://raw.githubusercontent.com/researchpy/Data-sets/master/insurance.csv")

df.head()

z = rp.summary_cont(df[['charges','age', 'children']])





dataset = pd.read_csv('/home/ehsan/Dropbox/junk/Machine Learning/code/Part 2 - Regression/Section 5 - Multiple Linear Regression/Multiple_Linear_Regression/50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

'''
columnTransformer = ColumnTransformer(
    [("States", OneHotEncoder(categories='auto'), [3])],
    remainder='passthrough')
'''

columnTransformer = make_column_transformer(
    (OneHotEncoder(categories='auto'), [3]),
    remainder='passthrough')

X = columnTransformer.fit_transform(X)

X = X[:, 1:]

Xtrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(Xtrain, YTrain)

YPred = regressor.predict(XTest)

X = np.append(arr = np.ones((50, 1)).astype(float), values = X, axis = 1)

XOpt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y.astype(float), exog = XOpt.astype(float), missing = 'drop').fit()
regressor_OLS.summary()

XOpt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y.astype(float), exog = XOpt.astype(float), missing = 'drop').fit()
regressor_OLS.summary()

XOpt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y.astype(float), exog = XOpt.astype(float), missing = 'drop').fit()
regressor_OLS.summary()

XOpt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = Y.astype(float), exog = XOpt.astype(float), missing = 'drop').fit()
regressor_OLS.summary()

XOpt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = Y.astype(float), exog = XOpt.astype(float), missing = 'drop').fit()
regressor_OLS.summary()


