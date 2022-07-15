import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom

dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values.astype(float)
y = dataset.iloc[:, -1].values.astype(int)

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

som = MiniSom(x=10, y=10, input_len=X.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_batch(data=X, num_iteration=1000)

plt.bone()
plt.pcolor(som.distance_map().T)
plt.colorbar()

markers = ['o', 's']
colors = ['red', 'green']

for i, x in enumerate(X):
    w = som.winner(x)
    plt.plot(w[0] + 0.5, w[1] + 0.5,
             markers[y[i]],
             markeredgecolor=colors[y[i]],
             markerfacecolor='None',
             markersize=15,
             markeredgewidth=2)

plt.show()

mappings = som.win_map(X)

distanceMap = som.distance_map()

frauds = []
for (x, y), value in np.ndenumerate(distanceMap):
    if distanceMap[x, y] > 0.7:
        print('x: {} y: {} mean distance: {}'.format(x, y, distanceMap[x, y]))
        inMapping = mappings[(x, y)]
        for inM in inMapping:
            frauds.append(inM)
            #print('n', inM)

frauds = np.array(frauds)
frauds = scaler.inverse_transform(frauds)
print('Possible fraud account numbers: {}'.format(frauds[:, 0]))

customer = dataset.iloc[:, 1:].values
isFraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    oo = dataset.iloc[i, 0].astype(float)
    if oo in frauds[:, 0]:
        isFraud[i] = 1



standardScalerX = MinMaxScaler(feature_range=(0, 1))
customer = standardScalerX.fit_transform(customer)

classifier = Sequential()

classifier.add(Dense(units=2, input_dim=X.shape[1], kernel_initializer='uniform', activation='relu'))

classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(customer, isFraud, batch_size=1, epochs=25)

y_pred = classifier.predict(customer)

y_pred = np.concatenate((dataset.iloc[:, 0:1].values.astype(float), y_pred), axis=1)
y_pred = y_pred[y_pred[:, 1].argsort()]

