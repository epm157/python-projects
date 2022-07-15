import gym
from gym import wrappers
import numpy as np


env = gym.make('CartPole-v0')

best_length = 0
episode_length = []
best_weights = np.zeros(4)

for i in range(100):
    new_weights = np.random.uniform(-1, 1, 4)
    length = []

    for j in range(100):
        observation = env.reset()
        counter = 0
        done = False

        while not done:
            counter += 1
            action = 1 if np.dot(observation, new_weights)>0 else 0
            #action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
        length.append(counter)

    average_length = float(sum(length) / len(length))
    episode_length.append(average_length)

    if average_length > best_length:
        best_length = average_length
        best_weights = new_weights

    if i % 10 == 0:
        print(i)

observation = env.reset()
done = False
counter = 0
while not done:
    env.render()
    counter += 1
    action = 1 if np.dot(observation, best_weights)>0 else 0
    observation, reward, done, info = env.step(action)

print(counter)

env.close()
