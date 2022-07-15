import numpy as np
import random
import torch as pt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


device = pt.device("cuda:0" if pt.cuda.is_available() else "cpu")
hidden_size = 50


class DinosDataset(Dataset):
    def __init__(self):
        super().__init__()
        with open('dinos.txt') as f:
            content = f.read().lower()
            self.vocab = sorted(set(content))
            self.vocab_size = len(self.vocab)
            self.lines = content.splitlines()
        self.ch_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_ch = {i: c for i, c in enumerate(self.vocab)}

    def __getitem__(self, index):
        line = self.lines[index]
        x_str = ' ' + line  # add a space at the beginning, which indicates a vector of zeros.
        y_str = line + '\n'
        x = pt.zeros([len(x_str), self.vocab_size], dtype=pt.float)
        y = pt.empty(len(x_str), dtype=pt.long)

        y[0] = self.ch_to_idx[y_str[0]]
        # we start from the second character because the first character of x was nothing(vector of zeros).
        for i, (x_ch, y_ch) in enumerate(zip(x_str[1:], y_str[1:]), 1):
            x[i][self.ch_to_idx[x_ch]] = 1
            y[i] = self.ch_to_idx[y_ch]

        return x, y

    def __len__(self):
        return len(self.lines)

trn_ds = DinosDataset()
trn_dl = DataLoader(trn_ds, batch_size=1, shuffle=True)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.linear_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.linear_u = nn.Linear(input_size + hidden_size, hidden_size)
        self.linear_c = nn.Linear(input_size + hidden_size, hidden_size)
        self.linear_o = nn.Linear(input_size + hidden_size, hidden_size)

        self.i2o = nn.Linear(hidden_size, output_size)

    def forward(self, c_prev, h_prev, x):
        combined = pt.cat([x, h_prev], 1)
        f = pt.sigmoid(self.linear_f(combined))
        u = pt.sigmoid(self.linear_u(combined))
        c_tilde = pt.tanh(self.linear_c(combined))
        c = f * c_prev + u * c_tilde
        o = pt.sigmoid(self.linear_o(combined))
        h = o * pt.tanh(c)
        y = self.i2o(h)

        return c, h, y

model = LSTM(trn_ds.vocab_size, hidden_size, trn_ds.vocab_size).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)


def print_sample(sample_idxs):
    print(trn_ds.idx_to_ch[sample_idxs[0]].upper(), end='')
    [print(trn_ds.idx_to_ch[x], end='') for x in sample_idxs[1:]]


def sample(model):
    model.eval()
    c_prev = pt.zeros([1, hidden_size], dtype=pt.float, device=device)
    h_prev = pt.zeros_like(c_prev)
    x = c_prev.new_zeros([1, trn_ds.vocab_size])
    sampled_indexes = []
    idx = -1
    n_chars = 1
    newline_char_idx = trn_ds.ch_to_idx['\n']
    with pt.no_grad():
        while n_chars != 50 and idx != newline_char_idx:
            c_prev, h_prev, y_pred = model(c_prev, h_prev, x)
            softmax_scores = pt.softmax(y_pred, 1).cpu().numpy().ravel()
            np.random.seed(np.random.randint(1, 5000))
            idx = np.random.choice(np.arange(trn_ds.vocab_size), p=softmax_scores)
            sampled_indexes.append(idx)

            x = (y_pred == y_pred.max(1)[0]).float()

            n_chars += 1

            if n_chars == 50:
                sampled_indexes.append(newline_char_idx)

    model.train()
    return sampled_indexes


def train_one_epoch(model, loss_fn, optimizer):
    model.train()
    for line_num, (x, y) in enumerate(trn_dl):
        loss = 0
        optimizer.zero_grad()
        c_prev = pt.zeros([1, hidden_size], dtype=pt.float, device=device)
        h_prev = pt.zeros_like(c_prev)
        x, y = x.to(device), y.to(device)
        for i in range(x.shape[1]):
            c_prev, h_prev, y_pred = model(c_prev, h_prev, x[:, i])
            loss += loss_fn(y_pred, y[:, i])

        if line_num % 100 == 0:
            print('Loss: {}'.format(loss.item()))

        if (line_num + 1) % 100 == 0:
            print_sample(sample(model))
        loss.backward()
        optimizer.step()


def train(model, loss_fn, optimizer, dataset='dinos', epochs=1):
    for e in range(1, epochs+1):
        print(f'{"-"*20} Epoch {e} {"-"*20}')
        train_one_epoch(model, loss_fn, optimizer)

train(model, loss_fn, optimizer, epochs=3)



