import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
from utils import plotLearning

class GenericNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions):
        super(GenericNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent:
    def __init__(self, alpha, beta, input_dims, gamma=0.99, n_actions=2,
                 layer1_size=256, layer2_size=256):
        self.gamma = gamma
        self.log_probs = None
        self.n_actions = n_actions
        self.actor = GenericNetwork(alpha, input_dims, layer1_size,
                                    layer2_size, n_actions=n_actions)
        self.critic = GenericNetwork(beta, input_dims, layer1_size,
                                     layer2_size, n_actions=1)

    def choose_action(self, observation):
        probabilities = self.actor.forward(observation)
        probabilities = F.softmax(probabilities)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.log_probs = log_probs
        return action.item()

    def learn(self, state, reward, new_state, done):

        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        new_critic_value = self.critic.forward(new_state)
        critic_value = self.critic.forward(state)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        delta = (reward+self.gamma*new_critic_value*(1-int(done)))-critic_value
        actor_loss = -self.log_probs * delta
        critic_loss = delta**2
        (actor_loss + critic_loss).backward()

        self.actor.optimizer.step()
        self.critic.optimizer.step()



if __name__ == '__main__':
    agent = Agent(alpha=0.00001, beta=0.00005, input_dims=[4], n_actions=2, gamma=0.99,
                  layer1_size=32, layer2_size=32)


    env = gym.make('CartPole-v1')
    score_history = []
    num_episodes = 30005
    for i in range(num_episodes):

        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            #action = np.array(action).reshape((1,))
            new_observation, reward, done, info = env.step(action)
            agent.learn(observation, reward, new_observation, done)
            observation = new_observation
            score += reward
        score_history.append(score)

        if i > 0 and i % 100 == 0:
            items = score_history[-100:]
            x = np.mean(items)
            #x = np.sum(score_history)
            #x = x / len(items)
            #y = np.mean(items)
            #print(str(x) + "    " + str(y))
            print('episode: ', i, 'avg score: %.2f' % x)

        if i > 0 and i % 1000 == 0:
            filename = 'cartpole' + str(i) + '.png'
            plotLearning(score_history, filename=filename, window=20)




