import gym
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        select = tf.random.categorical(logits, 1)
        return tf.squeeze(select, axis=-1)

class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super(Model, self).__init__()
        self.hidden1 = tf.keras.layers.Dense(128, activation='relu')
        self.hidden2 = tf.keras.layers.Dense(128, activation='relu')
        self.value = tf.keras.layers.Dense(1, name='value')
        self.logits = tf.keras.layers.Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution()

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        hidden_logs = self.hidden1(x)
        hidden_vals = self.hidden2(x)
        logs = self.logits(hidden_logs)
        vals = self.value(hidden_vals)

        return logs, vals

    def action_value(self, obs):
        logits, value = self.predict(obs)
        action = self.dist.predict(logits)
        action = np.squeeze(action, axis=-1)
        value = np.squeeze(value, axis=-1)
        return action, value


class A2CAgent:
    def __init__(self, model):
        self.params = {
            'gamma': 0.99,
            'value': 0.5,
            'entropy': 0.0001
        }
        self.model = model
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0007), loss=[self._logits_loss, self._value_loss])

    def train(self, env, batch_size=32, updates=1000):
        actions = np.empty((batch_size,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_size))
        observations = np.empty((batch_size,) + env.observation_space.shape)

        ep_rews = [0.0]

        next_obs = env.reset()
        for update in range(updates):
            for step in range(batch_size):
                observations[step] = next_obs.copy()
                actions[step], values[step] = self.model.action_value(next_obs[None, :])
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])

                ep_rews[-1] += rewards[step]
                if dones[step]:
                    ep_rews.append(0.0)
                    next_obs = env.reset()
                    logging.info("Episode: %03d, Reward: %03d" % (len(ep_rews)-1, ep_rews[-2]))

            _, next_value = self.model.action_value(next_obs[None, :])
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
            logging.debug("[%d/%d] Losses: %s" % (update + 1, updates, losses))

        return ep_rews

    def test(self, env, render=False):
        obs, done,ep_reward = env.reset(), False, 0
        while not done:
            action, _ = self.model.action_value(obs[None, :])
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
        return ep_reward

    def _returns_advantages(self, rewards, dones, values, next_value):
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t+1] * (1 - dones[t])
        returns = returns[:-1]
        advantages = returns - values
        return returns, advantages

    def _value_loss(self, returns, value):
        err = tf.keras.losses.mean_squared_error(returns, value)
        err = self.params['value'] * err
        return err

    def _logits_loss(self, acts_and_dvs, logits):
        actions, advantages = tf.split(acts_and_dvs, 2, axis=-1)
        weighted_sparse_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        entropy_loss = tf.keras.losses.categorical_crossentropy(logits, logits, from_logits=True)
        return policy_loss - self.params['entropy']*entropy_loss

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    env = gym.make('CartPole-v0')
    model = Model(num_actions=env.action_space.n)
    agent = A2CAgent(model)

    rewards_history = agent.train(env)
    print("Finished training.")
    print("Total Episode Reward: %d out of 200" % agent.test(env, True))

    plt.style.use('seaborn')
    plt.plot(np.arange(0, len(rewards_history), 25), rewards_history[::25])
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()































