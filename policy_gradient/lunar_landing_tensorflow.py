import tensorflow as tf
import numpy as np
from tensorflow import keras
import gym
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow_probability as tfp

def plotLearning(scores, filename, x=None, window=5):
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')
    plt.xlabel('Game')
    plt.plot(x, running_avg)
    plt.savefig(filename)


class PolicyGradientAgent():
    def __init__(self, ALPHA, GAMMA=0.95, n_actions=4, layer1_size=16, layer2_size=16, input_dims=128):

        self.lr = ALPHA
        self.gamma = GAMMA
        self.n_actons = n_actions
        self.action_space = [i for i in range(self.n_actons)]
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.input_dims = input_dims
        self.state_memory = []
        self.grads = []
        self.reward_memory = []
        self.model = self.build_net()

        self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)

        self.compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.gradBuffer = self.model.trainable_variables
        for ix, grad in enumerate(self.gradBuffer):
            self.gradBuffer[ix] = grad * 0


    def build_net(self):

        model = keras.Sequential()
        model.add(keras.layers.Dense(self.layer1_size, activation='relu', input_shape=[self.input_dims]))
        model.add(keras.layers.Dense(self.layer2_size, activation='relu'))
        model.add(keras.layers.Dense(self.n_actons, activation='softmax'))
        model.build()

        return model

    def choose_action(self, observation):

        observation = observation[np.newaxis, :]

        with tf.GradientTape() as tape:
            probabilities = self.model(observation)
            action_probs = tfp.distributions.Categorical(probabilities)
            action = action_probs.sample()
            log_probs = action_probs.log_prob(action)
            loss = self.compute_loss([action], probabilities)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.grads.append(grads)
        action = action.numpy()[0]

        '''
            logits = self.model(observation)
            action_dist = logits.numpy()
            # Choose random action with p = action dist
            action = np.random.choice(action_dist[0], p=action_dist[0])
            action = np.argmax(action_dist == action)
            loss = self.compute_loss([action], logits)
        grads = tape.gradient(loss, self.model.trainable_variables)

        self.grads.append(grads)
        '''



        return action


    def store_reward(self, reward):
        self.reward_memory.append(reward)


    def learn(self):

        reward_memory = np.array(self.reward_memory)

        G = np.zeros_like(reward_memory)

        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum

        mean = np.mean(G)
        std = np.std(G)
        if std == 0:
            std = 1

        G = (G - mean) / std


        for r, gr in zip(G, self.grads):
            for ix, grad in enumerate(gr):
                self.gradBuffer[ix] += grad * r

        self.optimizer.apply_gradients(zip(self.gradBuffer, self.model.trainable_variables))

        for ix, grad in enumerate(self.gradBuffer):
            self.gradBuffer[ix] = grad * 0

        self.grads = []
        self.reward_memory = []






if __name__ == '__main__':
    agent = PolicyGradientAgent(ALPHA=0.0005, input_dims=8, GAMMA=0.99,
                                n_actions=4, layer1_size=64, layer2_size=64)

    env = gym.make('LunarLander-v2')
    score_history = []
    score = 0
    num_episodes = 3000
    for i in range(num_episodes):
        print('episode: ', i,'score: ', score)
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_reward(reward)
            observation = observation_
            score += reward
        score_history.append(score)
        agent.learn()

        if i % 1000 == 0:
            filename = 'lunar-lander-tensorflow-' + str(i) + '.png'
            plotLearning(score_history, filename=filename, window=25)



















