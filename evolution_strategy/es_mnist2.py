import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing.dummy import Pool
from datetime import datetime

pool = Pool(4)

df = pd.read_csv('train.csv')
data = df.values.astype(np.float32)
np.random.shuffle(data)


X = data[:, 1:] / 255.0
Y = data[:, 0].astype(np.int32)

Xtrain = X[:-1000]
Ytrain = Y[:-1000]
Xtest = X[-1000:]
Ytest = Y[-1000:]

print(set(Y))

D = Xtrain.shape[1]
M = 100
K = len(set(Y))

def softmax(a):
    c = np.max(a, axis=1, keepdims=True)
    e = np.exp(a-c)
    result = e/ e.sum(axis=-1, keepdims=True)
    return result

def relu(x):
    result = x * (x>0)
    return result

def log_likelihood(Y, P):
    N = len(Y)
    result = np.log(P[np.arange(N), Y]).mean()
    return result


class ANN:
    def __init__(self, D, M, K):
        self.D = D
        self.M = M
        self.K = K

    def init(self):
        D, M, K = self.D, self.M, self.K
        self.W1 = np.random.randn(D, M) / np.sqrt(D)
        self.b1= np.zeros(M)
        self.W2 =np.random.randn(M, K) / np.sqrt(M)
        self.b2 = np.zeros(K)

    def forward(self, X):
        Z = X.dot(self.W1) + self.b1
        Z = np.tanh(Z)
        Z = Z.dot(self.W2) + self.b2
        Z = softmax(Z)
        return Z

    def score(self, X, Y):
        P = np.argmax(self.forward(X), axis=1)
        return np.mean(Y == P)

    def get_params(self):
        return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])

    def set_params(self, params):
        D, M, K = self.D, self.M, self.K
        self.W1 = params[:D * M].reshape(D, M)
        self.b1 = params[D * M:D * M + M]
        self.W2 = params[D * M + M:D * M + M + M * K].reshape(M, K)
        self.b2 = params[-K:]

def evolution_strategy(f, population_size, sigma, lr, initial_params, num_iters):
    num_params = len(initial_params)
    reward_per_iteration = np.zeros(num_iters)

    params = initial_params
    for t in range(num_iters):
        t0 = datetime.now()
        N = np.random.randn(population_size, num_params)
        R = pool.map(f, [params + sigma*N[j] for j in range(population_size)])
        R = np.array(R)

        m = R.mean()
        A = (R-m) / R.std()
        reward_per_iteration[t] = m
        params = params + lr / (population_size * sigma) * np.dot(N.T, A)
        print("Iter:", t, "Avg Reward:", m, "Duration:", (datetime.now() - t0))

    return params, reward_per_iteration

def reward_function(params):
    model = ANN(D, M, K)
    model.set_params(params)
    return model.score(Xtrain, Ytrain)


if __name__ == '__main__':
  model = ANN(D, M, K)
  model.init()
  params = model.get_params()
  best_params, rewards = evolution_strategy(
    f=reward_function,
    population_size=50,
    sigma=0.1,
    lr=0.2,
    initial_params=params,
    num_iters=600,
  )

  # plot the rewards per iteration
  plt.plot(rewards)
  plt.show()

  # final train and test accuracy
  model.set_params(best_params)
  print("Train score:", model.score(Xtrain, Ytrain))
  print("Test score:", model.score(Xtest, Ytest))

















