import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn import datasets

n_pts = 500
X, y = datasets.make_circles(n_samples=n_pts, random_state=123, noise=0.1, factor=0.2)
x_data = torch.Tensor(X)
y_data = torch.Tensor(y.reshape(-1, 1))

def scatter_plot():
  plt.scatter(X[y==0, 0], X[y==0, 1])
  plt.scatter(X[y==1, 0], X[y==1, 1])
  plt.show()

#scatter_plot()


class Model(nn.Module):
  def __init__(self, input_size, H1, output_size):
    super().__init__()
    self.linear1 = nn.Linear(input_size, H1)
    self.linear2 = nn.Linear(H1, output_size)

  def forward(self, x):
    x = torch.sigmoid(self.linear1(x))
    x = torch.sigmoid(self.linear2(x))
    return x

  def predict(self, x):
    y_pred = self.forward(x)
    if y_pred >= 0.5:
        return 1.0
    else:
        return 0.0

torch.manual_seed(2)
model = Model(2, 4, 1)
print(list(model.parameters()))


criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

epochs = 1000
losses = []

for i in range(epochs):
    y_pred = model.forward(x_data)
    loss = criterion(y_pred, y_data)
    losses.append(loss.item())
    if(len(losses) % 100 == 0):
        print("Epoch: {}, loss: {}".format(i, loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# plt.plot(range(epochs), losses)
# plt.ylabel('Loss')
# plt.xlabel('epoch')
# plt.show()


def plot_decision_boundary(X):
    x_span = np.linspace(min(X[:, 0]) - 0.25, max(X[:, 0]) + 0.25)
    y_span = np.linspace(min(X[:, 1]) - 0.25, max(X[:, 1]) + 0.25)
    xx, yy = np.meshgrid(x_span, y_span)
    xxr = xx.ravel()
    yyr = yy.ravel()
    grid = np.c_[xxr, yyr]
    grid = torch.Tensor(grid)
    pred_func = model.forward(grid)
    z = pred_func.view(xx.shape).detach().numpy()
    plt.contourf(xx, yy, z)


plot_decision_boundary(X)
scatter_plot()


x = 0.025
y = 0.025
point = torch.Tensor([x, y])
prediction = model.predict(point)
plt.plot([x], [y], marker='o', markersize=10, color='red')
print("Prediction is", prediction)
plot_decision_boundary(X)
plt.show()










