import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')

test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = int)

nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

def convert(data):
    new_data = []
    for i in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == i]
        id_rating = data[:, 2][data[:, 0] == i]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_rating
        shape = ratings.shape
        new_data.append(ratings)
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

trainingSetLen = len(training_set)
trainingSetSize = len(training_set[0])


training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

trainingSetLenTensor = len(training_set)
trainingSetSizeTensor = len(training_set[0])

class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

nb_epochs = 2000
for epoch in range(1, nb_epochs):
    trainLoss = 0
    s = 0.
    for id_user in range(nb_users):
        input = training_set[id_user]
        input = Variable(input)
        input = input.squeeze(0)
        target = input.clone()
        if torch.sum(input.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            meanCorrector = nb_movies/float(torch.sum(target > 0))
            trainLoss += np.sqrt(loss.data.item() * meanCorrector)
            loss.backward()
            s += 1.
            optimizer.step()

    print('epoch: ' + str(epoch) + ' loss: ' + str(trainLoss/s))

s = 0.
testLoss = 0.
for id_user in range(nb_users):
    input = training_set[id_user]
    input = Variable(input)
    input = input.squeeze(0)
    target = test_set[id_user]
    target = Variable(input)
    target = input.squeeze(0)
    if torch.sum(input.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        meanCorrector = nb_movies / (float(torch.sum(target > 0) + 1e-10))
        testLoss += np.sqrt(loss.data.item() * meanCorrector)

        s += 1.
        #optimizer.step()

print('test loss: ' + str(testLoss/s))




