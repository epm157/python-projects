import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from PIL import Image
from torch import nn
from torchvision import datasets, transforms, models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.RandomHorizontalFlip(),
                                      # transforms.RandomRotation(10),
                                      transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                      transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,), (0.5,))])

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
training_dataset = datasets.ImageFolder('data/ants_and_bees/train', transform=transform_train)
validation_dataset = datasets.ImageFolder('data/ants_and_bees/val', transform=transform)

training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=20, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=20, shuffle=False)


def img_convert(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
    image = image.clip(0, 1)
    return image


classes = ('ant', 'bee')

dataIter = iter(training_loader)
images, labels = dataIter.next()
fig = plt.figure(figsize=(25, 4))

for i in range(20):
    ax = fig.add_subplot(2, 10, i + 1, xticks=[], yticks=[])
    plt.imshow(img_convert(images[i]))
    ax.set_title(classes[labels[i].item()])

plt.show()

# model = models.alexnet(pretrained=True)
model = models.vgg16(pretrained=True)
print(model)

for params in model.features.parameters():
    params.requires_grad = False

n_inputs = model.classifier[6].in_features
last_layer = nn.Linear(n_inputs, len(classes))
model.classifier[6] = last_layer
model.to(device)
print(model.classifier[6].out_features)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 10
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
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels)

    else:

        epoch_loss = running_loss / len(training_loader.dataset)
        epoch_acc = running_corrects.float() / len(training_loader.dataset)
        running_loss_history.append(epoch_loss)
        running_corrects_history.append(epoch_acc.item())

        with torch.no_grad():
            for val_inputs, val_labels in validation_loader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)
                val_running_loss += val_loss

                _, val_preds = torch.max(val_outputs, 1)
                val_running_corrects += torch.sum(val_preds == val_labels)

            val_epoch_loss = val_running_loss / len(validation_loader.dataset)
            val_epoch_acc = val_running_corrects.float() / len(validation_loader.dataset)
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

url = 'http://cdn.sci-news.com/images/enlarge5/image_6425e-Giant-Red-Bull-Ant.jpg'
response = requests.get(url, stream=True)
img = Image.open(response.raw)
plt.imshow(img)
plt.show()

img = transform(img)
plt.imshow(img_convert(img))
plt.show()

image = img.to(device).unsqueeze(0)
output = model(image)
_, pred = torch.max(output, 1)
print(classes[pred.item()])

dataIter = iter(validation_loader)
images, labels = dataIter.next()
images = images.to(device)
labels = labels.to(device)
outputs = model(images)
_, preds = torch.max(outputs, 1)

fig = plt.figure(figsize=(25, 4))

for i in np.arange(20):
    ax = fig.add_subplot(2, 10, i + 1, xticks=[], yticks=[])
    plt.imshow(img_convert(images[i]))
    ax.set_title("{} ({})".format(str(classes[preds[i].item()]), str(classes[labels[i].item()])),
                 color=("green" if preds[i] == labels[i] else "red"))

plt.show()


