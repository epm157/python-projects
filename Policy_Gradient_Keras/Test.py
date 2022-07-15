from keras.layers import Dense, Activation, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
import numpy as np
from utils import plotLearning
import gym

class Agent:
    def __init__(self, ALPHA, GAMMA=0.99, n_actions=4, layer1_size=16, layer2_size=16, input_dims=128, fname='reinforce_2.h5'):

        self.lr = ALPHA
        self.gamma = GAMMA
        self.G = 0
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.policy, self.predict = self.build_policy_network()
        self.action_space = [i for i in range(self.n_actions)]
        self.model_file = fname

    def build_policy_network(self):
        input = tf.keras.layers.Input(shape=(self.input_dims,))
        advantage = tf.keras.layers.Input(shape=[1])
        dense1 = tf.keras.layers.Dense(self.fc1_dims, activation='relu')(input)
        dense2 = tf.keras.layers.Dense(self.fc2_dims, activation='relu')(dense1)
        probs = tf.keras.layers.Dense(self.n_actions, activation='softmax')(dense2)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1 - (1e-8))
            log_lik = y_true * K.log(out)
            return K.sum(-log_lik * advantage)

        policy = tf.keras.models.Model(inputs=[input, advantage], outputs=[probs])
        policy.compile(optimizer=tf.keras.optimizers.Adam(lr=self.lr), loss=custom_loss)

        predict = tf.keras.models.Model(inputs=[input], outputs=[probs])

        return policy, predict

    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.predict.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)
        return action

    def store_transition(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def learn(self):
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)

        actions = np.zeros([len(action_memory), self.n_actions])
        actions[(np.arange(len(action_memory)), action_memory)] = 1

        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            disount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k] * disount
                disount *= self.gamma
            G[t] = G_sum

        mean = np.mean(G)
        std = np.std(G)
        if std == 0:
            std = 1.0

        self.G = (G - mean) / std

        self.policy.train_on_batch([state_memory, self.G], actions)

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

    def save_model(self):
        self.policy.save(self.model_file)

if __name__ == '__main__':
    agent = Agent(ALPHA=0.0005, input_dims=8, GAMMA=0.99,
                  n_actions=4, layer1_size=64, layer2_size=64)

    env = gym.make('LunarLander-v2')
    score_history = []

    num_episodes = 2000

    for i in range(num_episodes):
        done = False
        score = 0
        state = env.reset()
        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, info = env.step(action)
            agent.store_transition(state, action,reward)
            state = new_state
            score += reward
        score_history.append(score)
        agent.learn()

        print('episode: ', i,'score: %.1f' % score,
            'average score %.1f' % np.mean(score_history[max(0, i-100):(i+1)]))

    filename = 'lunar-lander-keras.png'
    plotLearning(score_history, filename=filename, window=100)






