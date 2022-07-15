import torch
from torch.nn import Linear
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# def forward(x):
#     w = torch.tensor(3.0, requires_grad=True)
#     b = torch.tensor(1.0, requires_grad=True)
#     y = w*x + b
#     return y
#
# x = torch.tensor([4.0, 2.0])
# print(forward(x))
#
#
#
#
# model = Linear(in_features=1, out_features=1)
# print(model.bias)
# print(model.weight)
#
# x = torch.tensor([[2.0], [3.0]], requires_grad=True)
# print(model(x))




torch.manual_seed(1)

X = torch.randn(100, 1) * 10
y = X + 3*torch.randn(100, 1)

class LR(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super(LR, self).__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=output_size)

    def forward(self, x):
        pred = self.linear(x)
        return pred

model = LR(1, 1)

def get_params(mod):
    [w, b] = mod.parameters()
    return (w[0][0].item(), b[0].item())

def plot_fit(mod, title):
    plt.title = title
    w1, b1 = get_params(mod)
    x1 = np.array([-30, 30])
    y1 = w1*x1 + b1
    plt.plot(x1, y1, 'r')
    plt.scatter(X.numpy(), y.numpy())
    plt.show()

plot_fit(model, 'Initial Model')


criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 100
losses = []

for i in range(epochs):
    y_pred = model.forward(X)
    loss = criterion(y_pred, y)
    print("epoch:", i, "loss:", loss.item())

    losses.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.show()

plot_fit(model, 'Trained Model')


