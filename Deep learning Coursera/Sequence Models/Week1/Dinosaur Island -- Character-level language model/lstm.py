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
    def __init__(self, input_size, n_hidden, n_layers, drop_prob, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = n_layers
        self.lstm = nn.LSTM(input_size, n_hidden, n_layers, dropout=drop_prob, batch_first=True)
        #self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, output_size)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out


model = LSTM(trn_ds.vocab_size, hidden_size, 1, 0.2, trn_ds.vocab_size).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-2)

def print_sample(sample_idx):
    print(trn_ds.idx_to_ch[sample_idx[0]].upper(), end='')
    [print(trn_ds.idx_to_ch[x], end='') for x in sample_idx[1:]]

def sample(model):
    model.eval()
    #c_prev = torch.zeros([1, hidden_size], dtype=torch.float, device=device)
    #h_prev = torch.zeros_like(c_prev)
    #x = c_prev.new_zeros([1, trn_ds.vocab_size])

    x = torch.zeros([1, 1, trn_ds.vocab_size], dtype=torch.float, device=device)
    samples_indexes = []
    idx = -1
    n_chars = 1
    newline_char_idx = trn_ds.ch_to_idx['\n']
    with torch.no_grad():
        while n_chars < 50 and idx != newline_char_idx:
            y_pred = model(x)
            y_pred = y_pred.reshape(1, -1)
            softmax_scores = torch.softmax(y_pred, 1).cpu().numpy().ravel()
            np.random.seed(np.random.randint(1, 5000))
            idx = np.random.choice(np.arange(trn_ds.vocab_size), p=softmax_scores)
            samples_indexes.append(idx)

            x = (y_pred == y_pred.max(1)[0]).float()
            x = x.unsqueeze(0)
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

        #y_pred = model(x)
        #loss += loss_fn(y_pred, y)

        for i in range(x.shape[1]):
            input = x[:, i]
            input = input.unsqueeze(0)
            y_pred = model(input)
            y_pred = y_pred.reshape(1,-1)
            loss += loss_fn(y_pred, y[:, i])

        if line_num % 100 == 0:
            print('Loss: {}'.format(loss.item()))
        if (line_num + 1) % 100 == 0:
            print_sample(sample(model))


        loss.backward()
        optimizer.step()




