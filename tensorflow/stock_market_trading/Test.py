import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas_datareader as data_reader

from tqdm import tqdm_notebook, tqdm
from collections import deque


class AI_Trader:

    def __init__(self, state_size, action_space=3, model_name='AITrader'):
        self.state_size = state_size
        self.action_space = action_space

        self.memory = deque(maxlen=2000)
        self.inventory = []
        self.model_name = model_name

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995

        self.model = self.model_builder()

    def model_builder(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=32, activation='relu', input_dim=self.state_size))
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dense(units=self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001))
        return model

    def trade(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)
        action = self.model.predict(state)
        return np.argmax(action[0])


    def batch_train(self, batch_size):
        batch = []
        for i in range(len(self.memory) - batch_size + 1, len(self.memory)):
            batch.append(self.memory[i])

        for state, action, reward, next_state, done in batch:
            if not done:
                reward = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            target = self.model.predict(state)
            target[0][action] = reward

            self.model.fit(state, target, epoch=1, verbose=0)

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def stocks_price_format(n):
  if n < 0:
    return "- $ {0:2f}".format(abs(n))
  else:
    return "$ {0:2f}".format(abs(n))


def dataset_loader(stock_name):
    dataset = data_reader.DataReader(stock_name, data_source="yahoo")
    start_date = str(dataset.index[0]).split()[0]
    end_date = str(dataset.index[-1]).split()[0]
    close = dataset['close']
    return close


def state_creator(data, timestamp, window_size):
    starting_id = timestamp - window_size + 1
    if starting_id >= 0:
        windowed_data = data[starting_id: timestamp+1]
    else:
        windowed_data = -starting_id * [data[0]] + list(data[0: timestamp+1])

    state = []
    for i in range(window_size - 1):
        state.append(sigmoid(windowed_data[i+1] - windowed_data[i]))

    return np.array([state])

stock_name = "AAPL"
data = dataset_loader(stock_name)

window_size = 10
episodes = 1000

batch_size = 32
data_samples = len(data) - 1

trader = AI_Trader(window_size)
print(trader.model.summary())





















