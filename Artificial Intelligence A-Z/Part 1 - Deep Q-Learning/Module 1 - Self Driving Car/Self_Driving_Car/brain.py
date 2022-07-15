import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, input_size, nb_actions):
        self.input_size = input_size
        self.nb_actions = nb_actions
        self.fc1 = nn.Linear(self.input_size, 30)
        self.fc2 = nn.Linear(30, self.nb_actions)

    def forward(self, observation):
        x = self.fc1(observation)
        x = F.relu(x)
        q_vals = self.fc2(x)
        return q_vals

class ReplyMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        samples = map(lambda x: torch.Tensor(torch.cat(x, 0)), samples)
        return samples

a1 = (1, 2, 3, 4)
a2 = ('a', 'b', 'c', 'd')
a3 = ('one', 'two', 'three', 'four')
c = [a1, a2, a3]
d = zip(*c)
print(list(d))
