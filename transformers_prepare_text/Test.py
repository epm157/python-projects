import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


prefix = 'data/'


train_df = pd.read_csv(prefix + 'train.csv', header=None)
test_df = pd.read_csv(prefix + 'test.csv', header=None)

train_df[0] = (train_df[0] == 2).astype(int)
test_df[0] = (train_df[0] == 2).astype(int)

temp = ['a'] * train_df.shape[0]
print(len(temp))

temp = ['a'] * len(train_df)
print(len(temp))

train_df = pd.DataFrame({
    'id': range(len(train_df)),
    'label': train_df[0],
    'alpha': ['a'] * train_df.shape[0],
    'text': train_df[1].replace(r'\n', ' ', regex=True)
})

dev_df = pd.DataFrame({
    'id': range(len(test_df)),
    'label': test_df[0],
    'alpha': ['a'] * len(test_df),
    'text': test_df[1].replace(r'\n', ' ', regex=True)
})


train_df.to_csv('data/train2.tsv', sep='\t', index=False, header=False)
dev_df.to_csv('data/dev2.tsv', sep='\t', index=False, header=False)

print(train_df.head())


