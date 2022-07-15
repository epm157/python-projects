from simple_dqn2 import Agent
import numpy as np
import gym
from utils import plotLearning

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    lr = 0.0005
    n_games = 500
    agent = Agent(gamma=0.99, epsilon=0.0, alpha=lr, input_dims=8, n_actions=4, mem_size=1000000, batch_size=64, epsilon_end=0.0)

    scores = []
    eps_history = []

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            new_observation, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, new_observation, int(done))
            observation = new_observation
            agent.learn()

        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[max(0, i-00): (i+1)])
        print('episode: ', i, 'score: %.2f' % score,
              ' average score %.2f' % avg_score)

        if i % 10 == 0 and i > 0:
            agent.save_model()

    filename = 'lunarlander9.png'

    x = [i + 1 for i in range(n_games)]
    plotLearning(x, scores, eps_history, filename)










import numpy as np

t1 = np.array([1, 1, 1])
t2 = np.array([2, 3, 4])
t3 = np.array([6, 7, 8])


t4 = np.concatenate((t1, t2, t3), axis=0)
print(t4)

t5 = np.stack((t1, t2, t3), axis=0)
print(t5)

t6 = np.concatenate(
    (
        np.expand_dims(t1, 1),
        np.expand_dims(t2, 1),
        np.expand_dims(t3, 1),
    )
    , axis=1
)

print(t6)



import numpy as np

x = np.array([[0, 0, 1, 0], [0,1,0,0]])

action_values = np.array([0, 1, 2, 3], dtype=np.int8)
action_indices = np.dot(x, action_values)
print(action_indices)
