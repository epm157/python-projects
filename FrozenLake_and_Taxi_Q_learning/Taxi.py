import numpy as np
import gym
import random
import time


total_episodes = 50000        # Total episodes
total_test_episodes = 100     # Total test episodes
max_steps = 99                # Max steps per episode

learning_rate = 0.7           # Learning rate
gamma = 0.618                 # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.01             # Exponential decay rate for exploration prob





env = gym.make("Taxi-v2")
env.reset()
env.render()

action_size = env.action_space.n
print("Action size ", action_size)

state_size = env.observation_space.n
print("State size ", state_size)


qtable = np.zeros((state_size, action_size))
print(qtable)
print(qtable.shape)




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

for episode in range(5):
    state =env.reset()
    step = 0
    done = False
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):
        env.render()
        time.sleep(1)
        action = np.argmax(qtable[state])

        new_state, reward, done, info = env.step(action)

        if done:
            break
        state = new_state

    time.sleep(5)

env.close()









