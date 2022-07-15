import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values




Xtrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size = 1/3, random_state = 0)

regressor = LinearRegression()
regressor.fit(Xtrain, YTrain)

YPred = regressor.predict(XTest)


plt.scatter(Xtrain, YTrain, color = 'red')
plt.plot(Xtrain, regressor.predict(Xtrain), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(XTest, YTest, color = 'red')
plt.plot(Xtrain, regressor.predict(Xtrain), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
