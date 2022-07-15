import numpy as np
import gym
import random

total_episodes = 15000        # Total episodes
learning_rate = 0.8           # Learning rate
max_steps = 99                # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.005             # Exponential decay rate for exploration prob


env = gym.make("FrozenLake-v0")

action_size = env.action_space.n
state_size = env.observation_space.n

qtable = np.zeros((state_size, action_size))

print(qtable.shape)
print(qtable)




rewards = []

for episode in range(total_episodes):

    state = env.reset()
    step = 0
    one = False
    total_rewards = 0

    for step in range(max_steps):
        exp_exp_tradeoff = random.uniform(0, 1)

        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state, :])
        else:
            action = env.action_space.sample()


        new_state, reward, done, info = env.step(action)

        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

        total_rewards += reward

        state = new_state

        if done:
            break

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)

print ("Score over time: " +  str(sum(rewards)/total_episodes))
print(qtable)
print(epsilon)


env.reset()
env.render()
print(np.argmax(qtable,axis=1).reshape(4,4))

env.reset()

for episode in range(5):
    state =env.reset()
    step = 0
    done = False
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):

        env.render()
        action = np.argmax(qtable[state])

        new_state, reward, done, info = env.step(action)

        if done:
            break
        state = new_state

env.close()











