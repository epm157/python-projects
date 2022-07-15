import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import config
from utils import imshow, show_plot
from contrastive import ContrastiveLoss
import torchvision
from torch.autograd import Variable
from PIL import Image
import PIL.ImageOps
import os
import os.path
from os import path


base_dir = '/Users/ehsan/Dropbox/junk/ML/Siamese_Network_Signature/sign_data/'
training_dir = base_dir + "train"
training_csv = base_dir + "train_data.csv"
testing_csv = base_dir + "test_data.csv"
testing_dir = base_dir + "test"

#print(path.exists(testing_csv))

class SiameseDataset():
    def __init__(self, training_csv=None, training_dir=None, transform=None):
        self.train_df = pd.read_csv(training_csv)
        self.train_df.columns = ['image1', 'image2', 'label']
        self.train_dir = training_dir
        self.transform = transform

    def __getitem__(self, index):
        image1_path = os.path.join(self.train_dir, self.train_df.iat[index, 0])
        image2_path = os.path.join(self.train_dir, self.train_df.iat[index, 1])
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        img0 = img0.convert('L')
        img1 = img1.convert('L')

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, t.from_numpy(np.array([int(self.train_df.iat[index, 2])], dtype=np.float32))

    def __len__(self):
        return len(self.train_df)


trans = transforms.Compose([transforms.Resize((105, 105)), transforms.ToTensor()])
siamese_dataset = SiameseDataset(training_csv, training_dir, trans)


'''

# Viewing the sample of images and to check whether its loading properly
vis_dataloader = DataLoader(siamese_dataset, shuffle=True, batch_size=8)
dataiter = iter(vis_dataloader)
example_batch = next(dataiter)
concatenated = t.cat((example_batch[0], example_batch[1]), 0)
imshow(torchvision.utils.make_grid(concatenated))
print(example_batch[2].numpy())
'''

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3))

        self.fc1 = nn.Sequential(
            nn.Linear(30976, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),

            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 2))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

train_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=8, batch_size=config.batch_size)

net = SiameseNetwork()
criterion = ContrastiveLoss()
optimizer = t.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)

def train():
    loss = []
    counter = []
    iteration_number = 0
    for epoch in range(1, config.epochs):
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            print(f'i: {i}, loss: {loss_contrastive}')

        print("Epoch {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
        iteration_number += 10
        counter.append(iteration_number)
        loss.append(loss_contrastive.item())
    show_plot(counter, loss)
    return net


test_dataset = SiameseDataset(testing_csv, testing_dir, trans)
test_dataloader = DataLoader(test_dataset, num_workers=6, batch_size=1, shuffle=True)

def test():

    count = 0
    for i, data in enumerate(test_dataloader, 0):
        x0, x1, label = data
        concat = t.cat((x0, x1), 0)
        output1, output2 = model(x0, x1)
        eucledian_distance = F.pairwise_distance(output1, output2)
        if label == t.FloatTensor([[0]]):
            label = "Original Pair Of Signature"
        else:
            label = "Forged Pair Of Signature"

        imshow(torchvision.utils.make_grid(concat))
        print("Predicted Eucledian Distance:-", eucledian_distance.item())
        print("Actual Label:-", label)
        count = count + 1
        if count == 10:
            break


if __name__ == '__main__':
    model = train()
    t.save(model.state_dict(), 'model.pt')





