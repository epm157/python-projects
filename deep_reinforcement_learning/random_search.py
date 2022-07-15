import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt



def get_action(s, w):
    return 1 if s.dot(w) > 0 else 0

def play_one_episode(env, params):
    observation = env.reset()
    done = False
    t = 0
    while not done and t < 10000:
        t += 1
        action = get_action(observation, params)
        observation, reward, done, info = env.step(action)
        if done:
            break

    return t

def play_multiple_episodes(env, T, params):
    episode_length = np.empty(T)
    for i in range(T):
        episode_length[i] = play_one_episode(env, params)

    avg_length = episode_length.mean()
    print("avg length:", avg_length)
    return avg_length

def random_search(env):
    episode_length = []
    best = 0
    params = None
    for t in range(100):
        new_params = np.random.random(4)*2 - 1
        avg_length = play_multiple_episodes(env, 100, new_params)
        episode_length.append(avg_length)

        if avg_length > best:
            best = avg_length
            params = new_params

    return episode_length, params

if __name__ == '__main__':
  env = gym.make('CartPole-v0')
  episode_lengths, params = random_search(env)
  #plt.plot(episode_lengths)
  #plt.show()

  print("***Final run with final weights***")
  env = wrappers.Monitor(env, 'my_awesome_dir')
  play_multiple_episodes(env, 100, params)

