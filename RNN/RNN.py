import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split


import os
print(os.listdir("input"))

trainDf = pd.read_csv(r"input/train.csv",dtype = np.float32)

targets_numpy = trainDf.label.values
features_numpy = trainDf.loc[:,trainDf.columns != "label"].values/255


features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
                                                                             targets_numpy,
                                                                             test_size = 0.2,
                                                                             random_state = 42)


featureTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)

featureTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)


batch_size = 100
n_iters = 20000
num_epochs = n_iters / (len(features_train)/batch_size)
num_epochs = int(num_epochs)


trainDataset = torch.utils.data.TensorDataset(featureTrain, targetsTrain)
testDataset = torch.utils.data.TensorDataset(featureTest, targetsTest)

train_loader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=False)


# plt.imshow(features_numpy[10].reshape(28, 28))
# plt.axis('off')
# plt.show()


class RNNModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])

        return out


input_dim = 28
hidden_dim = 100
layer_dim = 2
output_dim = 10

model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)

error = nn.CrossEntropyLoss()

learning_rate = 0.005
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

seq_dim = 28
loss_list=[]
iteration_list=[]
accuracy_list=[]
count=0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        train = Variable(images.view(-1, seq_dim, input_dim))
        labels = Variable(labels)

        optimizer.zero_grad()

        outputs = model(train)

        loss = error(outputs, labels)

        loss.backward()

        optimizer.step()

        count += 1

        if count%250 == 0:
            correct = 0
            total = 0

            for images, labels in test_loader:
                images = Variable(images.view(-1, seq_dim, seq_dim))

                outputs = model(images)

                predicted = torch.max(outputs.data, 1)[1]

                total += labels.size(0)

                correct += (predicted == labels).sum()

            accuracy = 100 * correct / float(total)

            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)

            if count % 500 == 0:
                print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.item(), accuracy))



plt.plot(iteration_list,loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("RNN: Loss vs Number of iteration")
plt.show()

# visualization accuracy
plt.plot(iteration_list,accuracy_list,color = "red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("RNN: Accuracy vs Number of iteration")
plt.savefig('graph.png')
plt.show()




