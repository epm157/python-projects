
import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
import retro
import tensorflow as tf
from tensorflow import keras
import keras.backend.tensorflow_backend as backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam

#python3 -m retro.import ROMS/



from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames

import matplotlib.pyplot as plt # Display graphs

from collections import deque# Ordered collection with ends

import random

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')



env = retro.make(game='SpaceInvaders-Atari2600')

print("The size of our frame is: ", env.observation_space)
print("The action size is : ", env.action_space.n)

# Here we create an hot encoded version of our actions
# possible_actions = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0]...]

possible_actions = env.action_space.n
possible_actions = np.identity(possible_actions, dtype=int)
possible_actions = possible_actions.tolist()
possible_actions = np.array(possible_actions)

#possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())


print(possible_actions.shape)
print(possible_actions)


def preprocess_frame(frame):
    gray = rgb2gray(frame)
    cropped_frame = gray[8:-12, 4:-12]
    normalized_frame = cropped_frame / 255.0
    preprocessed_frame = transform.resize(normalized_frame, [110, 84])
    return preprocessed_frame


stack_size = 4

stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)
    if is_new_episode:
        stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for _ in range(stack_size)], maxlen=4)

        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
    else:
        stacked_frames.append(frame)

    stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames




state_size = [110, 84, 4]      # Our input is a stack of 4 frames hence 110x84x4 (Width, height, channels)
action_size = env.action_space.n # 8 possible actions
learning_rate =  0.00025      # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 500           # Total episodes for training
max_steps = 50000              # Max possible steps in an episode
batch_size = 64                # Batch size

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.00001           # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.9                    # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000          # Number of experiences the Memory can keep

### PREPROCESSING HYPERPARAMETERS
stack_size = 4                 # Number of frames stacked

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False

input_shape = state_size
action_space = env.action_space.n

class DQNetwork():

    def __init__(self, state_size, action_size, learning_rate):
        super(DQNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        self.model = self.create_model()


    def create_model(self):
        model = Sequential()

        model.add(Conv2D(256, (3, 3),
                         input_shape=(110, 84, 4)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))

        model.add(Dense(len(possible_actions), activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model


    def train(self, states, targets):
        t1 = states.shape
        t2 = targets.shape
        self.model.fit(states, targets, batch_size=batch_size)


DQNetwork = DQNetwork(state_size, action_size, learning_rate)


class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        indexes = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)

        return [self.buffer[i] for i in indexes]
    def size(self):
        size = len(self.buffer)
        return size


memory = Memory(max_size = memory_size)

'''


for i in range(pretrain_length):
    if i == 0:
        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    choice = np.random.randint(1, len(possible_actions)) - 1
    action = possible_actions[choice]
    next_state, reward, done, info = env.step(action)

    if done:
        next_state = np.zeros(state.shape)

        memory.add((state, action, reward, next_state, done))
        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    else:
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        memory.add((state, action, reward, next_state, done))
        state = next_state

'''



def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        choice = random.randint(0, len(possible_actions)-1)
        action = possible_actions[choice]
    else:
        qs = DQNetwork.model.predict(state)
        choice = np.argmax(qs)
        action = possible_actions[choice]
        print(action)


    return action, explore_probability





rewards_list = []

if training:
    decay_step = 0
    for episode in range(total_episodes):
        step = 0
        episode_rewards = []
        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)

        while step < max_steps:
            step += 1
            decay_step += 1
            action, explore_probability = predict_action(explore_start, explore_stop, decay_rate,
                                                         decay_step, state, possible_actions)
            next_state, reward, done, _ = env.step(action)

            if episode_render:
                env.render()

            episode_rewards.append(reward)

            if done:
                next_state = np.zeros((110, 84), dtype=np.int)
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                step = max_steps
                total_reward = np.sum(episode_rewards)

                print('Episode: {}'.format(episode),
                      'Total reward: {}'.format(total_reward),
                      'Explore P: {:.4f}'.format(explore_probability))

                rewards_list.append((episode, total_reward))
                memory.add((state, action, reward, next_state, done))

            else:
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                memory.add((state, action, reward, next_state, done))
                state = next_state


            if memory.size() >= batch_size:
                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []

                X = []
                y = []
                Qs_next_state = DQNetwork.model.predict(states_mb)

                for i, (current_state, action, reward, new_current_state, done) in enumerate(batch):
                    if dones_mb[i]:
                        target = rewards_mb[i]
                        #target_Qs_batch.append(rewards_mb[i])
                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])

                        #target_Qs_batch.append(target)

                    current_qs = Qs_next_state[i]
                    current_qs[action] = target
                    X.append(state)
                    y.append(current_qs)

                DQNetwork.train(np.array(X), np.array(y))
























