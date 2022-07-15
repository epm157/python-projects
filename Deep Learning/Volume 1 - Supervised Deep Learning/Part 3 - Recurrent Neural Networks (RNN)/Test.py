import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM

dataset = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset.iloc[:, 1:2].values.astype(float)

scaler = MinMaxScaler()
training_set = scaler.fit_transform(training_set)

XTrain = []
yTrain = []

for i in range(60, len(dataset)):
    XTrain.append(training_set[i-60: i, 0])
    yTrain.append(training_set[i, 0])

XTrain = np.array(XTrain)
yTrain = np.array(yTrain)

XTrain = np.reshape(XTrain, (XTrain.shape[0], XTrain.shape[1], 1))


regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (XTrain.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(XTrain, yTrain,  epochs = 300, batch_size = 32)


datasetTest = pd.read_csv('Google_Stock_Price_Test.csv')
realStockPrice = datasetTest.iloc[:, 1:2].values.astype(float)
totalDataset = pd.concat((dataset['Open'], datasetTest['Open']), axis = 0)
inputs = totalDataset[len(totalDataset) - len(datasetTest) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

XTest = []
for i in range(60, len(inputs)):
    XTest.append(inputs[i-60: i, 0])

XTest = np.array(XTest)
XTest = np.reshape(XTest, (XTest.shape[0], XTest.shape[1], 1))
predictedStockPrice = regressor.predict(XTest)
predictedStockPrice = scaler.inverse_transform(predictedStockPrice)

plt.plot(realStockPrice, color = 'red', label = 'Real Google Stock Price')
plt.plot(predictedStockPrice, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
