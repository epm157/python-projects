
import gym

env = gym.make('CartPole-v0')


env.reset()

box = env.observation_space
print(box)

print(env.action_space)


action = env.action_space.sample()
print(action)

observation, reward, done, info = env.step(action)

print(observation, reward, done, info)

done = False
while not done:
  observation, reward, done, info = env.step(env.action_space.sample())
  print(observation, reward, done, info)





