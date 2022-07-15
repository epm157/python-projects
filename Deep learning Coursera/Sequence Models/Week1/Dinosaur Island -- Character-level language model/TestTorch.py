import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hidden_size = 50

class DinosDataset(Dataset):
    def __init__(self):
        super().__init__()
        with open('/home/ehsan/Dropbox/junk/ML/Deep learning Coursera/Sequence Models/Week1/Dinosaur Island -- Character-level language model/dinos.txt') as f:
            content = f.read().lower()
            self.vocab = sorted(set(content))
            self.vocab_size = len(self.vocab)
            self.lines = content.splitlines()
        self.ch_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_ch = {i: c for i, c in enumerate(self.vocab)}

    def __getitem__(self, index):
        line = self.lines[index]
        x_str = ' ' + line
        y_str = line + '\n'
        x = torch.zeros([len(x_str), self.vocab_size], dtype=torch.float)
        y = torch.empty(len(x_str), dtype=torch.long)

        y[0] = self.ch_to_idx[y_str[0]]
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
        combined = torch.cat([x, h_prev], 1)
        f = torch.sigmoid(self.linear_f(combined))
        u = torch.sigmoid(self.linear_u(combined))
        c_tilde = torch.tanh(self.linear_c(combined))
        c = f * c_prev + u * c_tilde
        o = torch.sigmoid((self.linear_o(combined)))
        h = o * torch.tanh(c)
        y = self.i2o(h)

        return c, h, y
model = LSTM(trn_ds.vocab_size, hidden_size, trn_ds.vocab_size).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-2)

def print_sample(sample_idx):
    print(trn_ds.idx_to_ch[sample_idx[0]].upper(), end='')
    [print(trn_ds.idx_to_ch[x], end='') for x in sample_idx[1:]]

def sample(model):
    model.eval()
    c_prev = torch.zeros([1, hidden_size], dtype=torch.float, device=device)
    h_prev = torch.zeros_like(c_prev)
    x = c_prev.new_zeros([1, trn_ds.vocab_size])
    samples_indexes = []
    idx = -1
    n_chars = 1
    newline_char_idx = trn_ds.ch_to_idx['\n']
    with torch.no_grad():
        while n_chars < 50 and idx != newline_char_idx:
            c_prev, h_prev, y_pred = model(c_prev, h_prev, x)
            softmax_scores = torch.softmax(y_pred, 1).cpu().numpy().ravel()
            np.random.seed(np.random.randint(1, 5000))
            idx = np.random.choice(np.arange(trn_ds.vocab_size), p=softmax_scores)
            samples_indexes.append(idx)

            x = (y_pred == y_pred.max(1)[0]).float()
            n_chars +=1

            if(n_chars == 50):
                samples_indexes.append(newline_char_idx)
    model.train()
    return samples_indexes



epochs=3
for e in range(1, epochs + 1):
    print(f'{"-" * 20} Epoch {e} {"-" * 20}')
    model.train()
    for line_num, (x, y) in enumerate(trn_dl):
        loss = 0
        optimizer.zero_grad()
        c_prev = torch.zeros([1, hidden_size], dtype=torch.float, device=device)
        h_prev = torch.zeros_like(c_prev)
        x = x.to(device)
        y = y.to(device)

        for i in range(x.shape[1]):
            c_prev, h_prev, y_pred = model(c_prev, h_prev, x[:, i])
            loss += loss_fn(y_pred, y[:, i])

        if line_num % 100 == 0:
            print('Loss: {}'.format(loss.item()))

        if (line_num + 1) % 100 == 0:
            print_sample(sample(model))

        loss.backward()
        optimizer.step()




