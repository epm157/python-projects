import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
import PIL.ImageOps
import requests
from PIL import Image


transform = transforms.Compose([transforms.Resize((28, 28)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,))])
training_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
validation_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=100, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=100, shuffle=False)

def img_convert(tensor):
    image = tensor.clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array([0.5, 0.5, 0.5]) +np.array([0.5, 0.5, 0.5])
    image = image.clip(0, 1)
    return image

dataIter = iter(training_loader)
images, labels = dataIter.next()
fig = plt.figure(figsize=(25, 4))

for i in range(20):
    ax = fig.add_subplot(2, 10, i+1, xticks=[], yticks=[])
    plt.imshow(img_convert(images[i]))
    ax.set_title(labels[i].item())

plt.show()


class Classifier(nn.Module):

    def __init__(self, d_in, h1, h2, d_out):
        super().__init__()
        self.linear1 = nn.Linear(d_in, h1)
        self.dropout = nn.Dropout(0.3)
        self.linear2 = nn.Linear(h1, h2)
        self.linear3 = nn.Linear(h2, d_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.linear3(x)
        return x

model = Classifier(784, 125, 65, 10)
print(model)


criterion = nn. CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 20
running_loss_history = []
running_corrects_history = []
val_running_loss_history = []
val_running_corrects_history = []

for e in range(epochs):

    running_loss = 0.0
    running_corrects = 0.0

    val_running_loss = 0.0
    val_running_corrects = 0.0

    for inputs, labels in training_loader:
        inputs = inputs.view(inputs.shape[0], -1)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels)

    else:

        epoch_loss = running_loss / len(training_loader)
        epoch_acc = running_corrects.float() / len(training_loader)
        running_loss_history.append(epoch_loss)
        running_corrects_history.append(epoch_acc.item())

        with torch.no_grad():
            for val_inputs, val_labels in validation_loader:
                val_inputs = val_inputs.view(val_inputs.shape[0], -1)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)
                val_running_loss += val_loss

                _, val_preds = torch.max(val_outputs, 1)
                val_running_corrects += torch.sum(val_preds == val_labels)

            val_epoch_loss = val_running_loss/len(validation_loader)
            val_epoch_acc = val_running_corrects.float()/len(validation_loader)
            val_running_loss_history.append(val_epoch_loss)
            val_running_corrects_history.append(val_epoch_acc)

    print('Epoch: {}'.format(e + 1))
    print('Training loss: {:.4f}, Accuracy: {:.4f}'.format(epoch_loss, epoch_acc.item()))
    print('Validation loss: {:.4f}, Accuracy: {:.4f}'.format(val_epoch_loss, val_epoch_acc.item()))





plt.plot(running_loss_history, label='Training loss')
plt.plot(val_running_loss_history, label='Validation loss')
plt.show()


plt.plot(running_corrects_history, label='Training accuracy')
plt.plot(val_running_corrects_history, label='Validation accuracy')
plt.legend()
plt.show()


url = 'https://images.homedepot-static.com/productImages/007164ea-d47e-4f66-8d8c-fd9f621984a2/svn/architectural-mailboxes-house-letters-numbers-3585b-5-64_1000.jpg'
response = requests.get(url, stream=True)
img = Image.open(response.raw)
plt.imshow(img)
plt.show()


img = PIL.ImageOps.invert(img)
img.convert('1')
img = transform(img)
plt.imshow(img_convert(img))
plt.show()


img = img.view(img.shape[0], -1)
output = model(img)
_, pred = torch.max(output, 1)
print(pred)



dataIter = iter(validation_loader)
images, labels = dataIter.next()
images_ = images.view([images.shape[0], -1])
outputs = model(images_)
_, preds = torch.max(outputs, 1)

fig = plt.figure(figsize=(25, 4))

for i in np.arange(20):
    ax = fig.add_subplot(2, 10, i + 1, xticks=[], yticks=[])
    plt.imshow(img_convert(images[i]))
    ax.set_title("{} ({})".format(str(preds[i].item()), str(labels[i].item())), color=("green" if preds[i]==labels[i] else "red"))

plt.show()
