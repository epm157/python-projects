import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

import gym
from simple_dqn_torch import DeepQNetwork, Agent
from utils import plotLearning
import numpy as np
from gym import wrappers



class DeepQNetwork(nn.Module):
    def __init__(self, ALPHA, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        state = T.Tensor(observation).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions


class Agent():
    def __init__(self, gamma, epsilon, alpha, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.01, eps_dec=0.996):
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_MIN = eps_end
        self.EPS_DEC = eps_dec
        self.ALPHA = alpha
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_counter = 0
        self.Q_eval = DeepQNetwork(alpha, n_actions=self.n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)
        self.state_memory = np.zeros((self.mem_size, *input_dims))
        self.new_state_memory = np.zeros((self.mem_size, *input_dims))
        #self.action_memory = np.zeros((self.mem_size, self.n_actions), dtype=np.uint8)
        self.action_memory = np.zeros(self.mem_size, dtype=np.uint8)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def storeTransition(self, state, action, reward, new_state, terminal):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        #actions = np.zeros(self.n_actions)
        #actions[action] = 1.0
        #self.action_memory[index] = actions
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.terminal_memory[index] = 1 - terminal
        self.mem_counter += 1

    def chooseAction(self, observation):
        rand = np.random.random()
        actions = self.Q_eval.forward(observation)
        if rand > self.EPSILON:
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.mem_counter > self.batch_size:

            self.Q_eval.optimizer.zero_grad()

            max_mem = min(self.mem_counter, self.mem_size)

            batch = np.random.choice(max_mem, self.batch_size)
            state_batch = self.state_memory[batch]

            #action_batch = self.action_memory[batch]
            #action_values = np.array(self.action_space, dtype=np.uint8)
            #action_indices = np.dot(action_batch, action_values)

            action_indices = self.action_memory[batch]

            reward_batch = self.reward_memory[batch]
            new_state_batch = self.new_state_memory[batch]
            terminal_batch = self.terminal_memory[batch]

            reward_batch = T.Tensor(reward_batch).to(self.Q_eval.device)
            terminal_batch = T.Tensor(terminal_batch).to(self.Q_eval.device)

            q_eval = self.Q_eval.forward(state_batch).to(self.Q_eval.device)
            q_target = self.Q_eval.forward(state_batch).to(self.Q_eval.device)
            q_next = self.Q_eval.forward(new_state_batch).to(self.Q_eval.device)

            batch_index = np.arange(self.batch_size, dtype=np.int32)
            q_target[batch_index, action_indices] = reward_batch + self.GAMMA*T.max(q_next, dim=1)[0]*terminal_batch

            self.EPSILON = self.EPSILON * self.EPS_DEC if self.EPSILON > self.EPS_MIN else self.EPS_MIN

            loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
            loss.backward()
            self.Q_eval.optimizer.step()






def main():
    env = gym.make('LunarLander-v2')
    brain = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
                  input_dims=[8], alpha=0.003)

    scores = []
    eps_history = []
    num_games = 500
    score = 0


    for i in range(num_games):
        if i > 0 and i % 10 == 0:
            avg_score = np.mean(scores[max(0, i-10): (i+1)])
            print('episode: ', i, 'score: ', score,
                  ' average score %.3f' % avg_score,
                  'epsilon %.3f' % brain.EPSILON)
        else:
            print('episode: ', i, 'score: ', score)

        eps_history.append(brain.EPSILON)

        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = brain.chooseAction(observation)
            new_observation, reward, done, info = env.step(action)
            brain.storeTransition(observation, action, reward, new_observation, done)
            score += reward
            observation = new_observation
            brain.learn()

        scores.append(score)


    x = [i+1 for i in range(num_games)]
    filename = 'result' + str(int(time.time())) + '.png'
    plotLearning(x, scores, eps_history, filename)


if __name__ == '__main__':
    main()

print('result' + str(int(time.time())) + '.png')



