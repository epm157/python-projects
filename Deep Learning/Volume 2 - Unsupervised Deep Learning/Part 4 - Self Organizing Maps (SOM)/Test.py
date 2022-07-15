import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    '''
    w0 = w[0]
    w1 = w[1]
    wy = y[i]
    if w0 == 1 and w1 == 6:
        if y[i] == 1:
            str = "y[i] is 1"
        else:
            str = "y[i] is 0"
    '''
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
    if distanceMap[x, y] > 0.9:
        print('x: {} y: {} mean distance: {}'.format(x, y, distanceMap[x, y]))
        inMapping = mappings[(x, y)]
        for inM in inMapping:
            frauds.append(inM)
            print('n', inM)

frauds = np.array(frauds)
#frauds = np.concatenate((mappings[(8, 1)], mappings[(6, 8)]), axis=0)
frauds = scaler.inverse_transform(frauds)

print('Possible fraud account numbers: {}'.format(frauds[:, 0]))

'''
print(mappings.keys())
print(mappings['({},{})'.format(7, 0)])
print(mappings[(9, 9)])
'''

