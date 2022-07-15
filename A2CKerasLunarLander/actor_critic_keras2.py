import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

class Agent:
    def __init__(self, alpha, beta, gamma=0.99, n_actions=4,
                 layer1_size=1024, layer2_size=512, input_dims=8):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions

        self.actor, self.critic, self.policy = self.build_actor_critic_network()
        self.action_space = [i for i in range(n_actions)]

    def build_actor_critic_network(self):
        input = Input(shape=(self.input_dims,))
        delta = Input(shape=[1])
        dense1 = Dense(self.fc1_dims, activation='relu')(input)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        probs = Dense(self.n_actions, activation='softmax')(dense2)
        values = Dense(1, activation='linear')(dense2)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-(1e-8))
            log_lik = y_true * K.log(out)
            return K.sum(-log_lik * delta)

        actor = Model(inputs=[input, delta], outputs=[probs])
        actor.compile(optimizer=Adam(lr=self.alpha), loss=custom_loss)

        critic = Model(inputs=[input], outputs=[values])
        critic.compile(optimizer=Adam(lr=self.beta), loss='mean_squared_error')

        policy = Model(inputs=[input], outputs=[probs])

        return actor, critic, policy

    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        proabilities = self.policy.predict(state)[0]
        action = np.random.choice(self.action_space, p=proabilities)
        return action

    def learn(self, state, action, reward, new_state, done):
        state = state[np.newaxis, :]
        new_state = new_state[np.newaxis, :]
        critic_value_new = self.critic.predict(new_state)
        critic_value = self.critic.predict(state)

        target = reward + self.gamma * critic_value_new * (1-int(done))
        delta = target - critic_value

        actions = np.zeros([1, self.n_actions])
        actions[np.arange(1), action] = 1

        self.actor.fit([state, delta], action, verbose=0)
        self.critic.fit(state, target, verbose=0)
