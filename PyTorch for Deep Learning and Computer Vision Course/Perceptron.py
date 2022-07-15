import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn import datasets



n_pts = 100
centers = [[-0.5, 0.5], [0.5, -0.5]]

X,y = datasets.make_blobs(n_samples=n_pts, random_state=123, centers=centers, cluster_std=0.4)

def scatter_plot():
    plt.scatter(X[y==0, 0], X[y==0, 1])
    plt.scatter(X[y==1, 0], X[y==1, 1])
    plt.show()


x_data = torch.Tensor(X).float()
y_data = torch.Tensor(y.reshape(100, 1)).float()

class Model(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super().__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=output_size)

    def forward(self, x):
        pred = torch.sigmoid(self.linear(x))
        return pred

    def predict(self, x):
        y_pred = self.forward(x)
        if y_pred >= 0.5:
            return 1.0
        else:
            return 0.0

torch.manual_seed(2)
model = Model(2, 1)

print(list(model.parameters()))

[w, b] = model.parameters()
print(w, b)

w1, w2 = w.view(2)
b1 = b[0]
print(w1.item(), w2.item(), b.item())

def get_params():
    return (w1.item(), w2.item(), b.item())

def plot_fit(title):
    plt.title = title
    w1, w2, b1 = get_params()
    x1 = np.array([-2, 2])
    x2 = (w1*x1 + b1)/-w2
    plt.plot(x1, x2, 'r')
    scatter_plot()
    plt.show()

plot_fit('Initial Model')

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


epochs = 10000
losses = []

for i in range(epochs):
    y_pred = model.forward(x_data)
    loss = criterion(y_pred, y_data)
    print("epoch:", i, "loss:", loss.item())
    losses.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.grid()
plt.show()



point1 = torch.Tensor([1.0, -1.0])
point2 = torch.Tensor([-1.0, 1.0])

plt.plot(point1.numpy()[0], point1.numpy()[1], 'ro')
plt.plot(point2.numpy()[0], point2.numpy()[1], 'ko')

plot_fit('Trained Model')


print("Red point positive probability = {}".format(model.forward(point1).item()))
print("Black point positive probability = {}".format(model.forward(point2).item()))

print("Red point positive probability = {}".format(model.predict(point1)))
print("Black point positive probability = {}".format(model.predict(point2)))












