import torch as T
from model import DeepQNetwork
import gym
import time

import matplotlib.pyplot as plt
import numpy as np

def plotLearning(x, scores, epsilons, filename):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-5):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    plt.savefig(filename)


class Agent(object):
    def __init__(self, gamma, epsilon, alpha, max_memory_size, eps_end=0.05, replace=10_000, action_space=[0, 1, 2, 3, 4, 5]):
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_END = eps_end
        self.action_space = action_space
        self.mem_size = max_memory_size
        self.steps = 0
        self.learn_step_counter = 0
        self.memory = []
        self.mem_counter = 0
        self.replace_target_counter = replace
        self.Q_eval = DeepQNetwork(alpha)
        self.Q_next = DeepQNetwork(alpha)

    def store_transition(self, state, action, reward, state_):
        if self.mem_counter < self.mem_size:
            self.memory.append([state, action, reward, state_])
        else:
            self.memory[self.mem_counter%self.mem_size] = [state, action, reward, state_]
        self.mem_counter += 1

    def choose_action(self, observation):

        rand = np.random.random()
        actions = self.Q_eval.forward(observation)
        if rand < 1 - self.EPSILON:
            action = T.argmax(actions[1]).item()
        else:
            action = np.random.choice(self.action_space)

        self.steps += 1
        return action

    def learn(self, batch_size):
        self.Q_eval.zero_grad()

        if self.replace_target_counter is not None and \
            self.learn_step_counter % self.replace_target_counte == 0:

            self.Q_next.load_state_dict(self.Q_eval.state_dict())

        if self.mem_counter+batch_size < self.mem_size:
            memStart = int(np.random.choice(range(self.mem_counter)))
        else:
            memStart = int(np.random.choice(range(self.mem_size-batch_size)))

        miniBatch = self.memory[memStart:memStart+batch_size]
        memory = np.array(miniBatch)

        Q_pred = self.Q_eval.forward(list(memory[:, 0][:])).to(self.Q_eval.device)
        Q_next = self.Q_next.forward(list(memory[:, 3][:])).to(self.Q_eval.device)

        maxA = T.argmax(Q_next, dim=1).to(self.Q_eval.device)
        rewards = T.Tensor(list(memory[:, 2][:])).to(self.Q_eval.device)
        Q_target = Q_pred
        indices = np.arange(batch_size)
        Q_target[indices, maxA] = rewards + self.GAMMA*T.max(Q_next[1])

        if self.steps > 500:
            if self.EPSILON - 1e-4 > self.EPS_END:
                self.EPSILON -= 1e-4
            else:
                self.EPSILON = self.EPS_END

        loss = self.Q_eval.loss(Q_target, Q_pred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1


if __name__ == '__main__':
    env = gym.make('SpaceInvaders-v0')
    brain = Agent(gamma=0.95, epsilon=1.0,
                  alpha=0.003, max_memory_size=5000,
                  replace=None)

while brain.mem_counter < brain.mem_size:
    observation = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        new_observation, reward, done, info = env.step(action)
        if done and info['ale.lives'] == 0:
            reward = -100
        brain.store_transition(np.mean(observation[15:200, 30:125], axis=2),
                               action, reward,
                               np.mean(new_observation[15:200, 30:125], axis=2))
        observation = new_observation
    print('done initializing memory')

    scores = []
    epsHistory = []
    numGames = 50
    batch_size = 32

    for i in range(numGames):
        print('starting game ', i + 1, 'epsilon: %.4f' % brain.EPSILON)
        epsHistory.append(brain.EPSILON)
        done = False
        observation = env.reset()
        frames = [np.sum(observation[15:200, 30:125], axis=2)]
        score = 0
        lastAction = 0

        while not done:
            if len(frames) == 3:
                action = brain.choose_action(frames)
                frames = []
            else:
                action = lastAction
            new_observation, reward, done, info = env.step(action)
            score += reward
            frames.append(np.sum(new_observation[15:200, 30:125], axis=2))
            if done and info['ale.lives'] == 0:
                reward = -100
            brain.store_transition(np.mean(observation[15:200, 30:125], axis=2),
                                   action, reward,
                                   np.mean(new_observation[15:200, 30:125], axis=2))
            observation = new_observation
            brain.learn(batch_size)
            lastAction = action
        scores.append(score)
        print('score:', score)

    x = [i + 1 for i in range(numGames)]
    fileName = 'result' + str(time.time()) + '.png'
    plotLearning(x, scores, epsHistory, fileName)





