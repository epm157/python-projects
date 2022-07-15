import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor




x = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
z = np.array(x)
z.shape
(3, 4)
z.reshape(-1)

z.reshape(-1, 4)





# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


regressor  = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)


y_pred = regressor.predict(np.array([[6.5]]).reshape(1, -1))




X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()