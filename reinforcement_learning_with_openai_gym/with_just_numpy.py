import numpy as np
import gym
import matplotlib.pyplot as plt
from time import sleep


ps_space = np.linspace(-1.2, 0.6, 20)

vel_space = np.linspace(-0.07, 0.07, 20)

def max_action(Q, state, actions=[0, 1, 2]):

    # l1 = []
    # for a in actions:
    #     t1 = Q[(state, a)]
    #     l1.append(t1)

    values = np.array([Q[(state, a)] for a in actions])
    action = np.argmax(values)

    return action



def get_state(observation):
    pos, vel = observation
    pos_bin = np.digitize(pos, ps_space)
    vel_bin = np.digitize(vel, vel_space)

    pos_bin = pos_bin.item()
    vel_bin = vel_bin.item()
    return (pos_bin, vel_bin)



if __name__ == '__main__':

    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 1000
    n_games = 500000
    alpha = 0.1
    gamma = 0.99
    eps = 1.0

    states = []
    for pos in range(21):
        for vel in range(21):
            states.append((pos, vel))

    Q = {}
    for state in states:
        for action in [0, 1, 2]:
            Q[state, action] = 0


    score = 0
    total_rewards = np.zeros(n_games)
    for i in range(n_games):
        done = False
        obs = env.reset()
        state = get_state(obs)
        #print(obs, state)
        if i % 1000 == 0 and i > 0:
            print('episode ', i, ' score ', score, ' epsilon ', eps)

        score = 0

        while not done:
            #action = env.action_space.sample()
            action = np.random.choice([0, 1, 2]) if np.random.random() < eps else max_action(Q, state)
            obs_, reward, done, info = env.step(action)
            state_ = get_state(obs_)
            score += reward
            #print(obs_, state_)
            #input()
            #sleep(0.5)

            action_ = max_action(Q, state_)
            Q[state, action] = Q[state, action] + alpha*(reward + gamma*Q[state_, action_] - Q[state, action])

            state = state_

        total_rewards[i] = score
        eps = max(eps - 2/n_games, 0.01)


        if i%10000 == 0 and i > 0:
            mean_rewards = np.zeros(i)
            for t in range(i):
                mean_rewards[t] = np.mean(total_rewards[max(0, t - 50): (t + 1)])
            plt.plot(mean_rewards)
            plt.show()




    mean_rewards = np.zeros(n_games)
    for t in range(n_games):
        mean_rewards[t] = np.mean(total_rewards[max(0, t-50): (t+1)])
    plt.plot(mean_rewards)
    #plt.show()
    plt.savefig('with_just_numpy_result.pdf')
