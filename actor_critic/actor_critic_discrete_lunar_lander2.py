import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
from utils import plotLearning

class ActorCriticNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.pi = nn.Linear(self.fc2_dims, n_actions)
        self.v = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        state = T.Tensor(observation).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        pi = self.pi(x)
        v = self.v(x)

        return (pi, v)


class NewAgent:
    def __init__(self, alpha, input_dims, gamma=0.99, layer1_size=256, layer2_size=256, n_actions=2):
        self.gamma = gamma
        self.actor_critic = ActorCriticNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions=n_actions)
        self.log_probs=None

    def choose_action(self, observation):
        probabilities, _ = self.actor_critic.forward(observation)
        probabilities = F.softmax(probabilities)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.log_probs = log_probs
        return action.item()

    def learn(self, state, reward, new_state, done):

        self.actor_critic.optimizer.zero_grad()

        _, new_critic_value = self.actor_critic.forward(new_state)
        _, critic_value = self.actor_critic.forward(state)
        reward = T.tensor(reward, dtype=T.float).to(self.actor_critic.device)

        if done:
            delta = reward - critic_value
        else:
            delta = reward + self.gamma*new_critic_value - critic_value

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2


        (actor_loss + critic_loss).backward()
        self.actor_critic.optimizer.step()



if __name__ == '__main__':
    agent = NewAgent(alpha=0.000005, input_dims=[8], gamma=0.99, n_actions=4,
                  layer1_size=2048, layer2_size=512)

    env = gym.make('LunarLander-v2')
    score_history = []
    num_episodes = 2005
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
        print('episode: ', i, 'score: %.2f' % score)
        if i % 100 == 0:
            #filename = 'LunarLander-actor-critic-' + str(i) +'.png'
            filename = 'LunarLander-actor-critic-4-' + str(i) + '.png'
            plotLearning(score_history, filename=filename, window=20)
