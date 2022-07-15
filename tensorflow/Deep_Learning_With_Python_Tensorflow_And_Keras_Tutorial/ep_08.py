import pandas as pd
from collections import deque
from sklearn import preprocessing
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

import time

df = pd.read_csv('crypto_data/LTC-USD.csv', names = ['time', 'low', 'high', 'open', 'close', 'volume'])
print(df.head())

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = 'LTC-USD'
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"


def classify(current, future):
    if float(future) > float(current):
        return 1
    return 0

main_df = pd.DataFrame()

ratios = ['BTC-USD', 'LTC-USD', 'ETH-USD', 'BCH-USD']

for ratio in ratios:
    dataset = f'crypto_data/{ratio}.csv'
    df = pd.read_csv(dataset, names = ['time', 'low', 'high', 'open', 'close', 'volume'])
    #print(df.head())
    df.rename(columns={'close': f'{ratio}_close', 'volume': f'{ratio}_volume'}, inplace=True)

    df.set_index('time', inplace=True)
    df = df[[f'{ratio}_close', f'{ratio}_volume']]
    print(df.head())

    if main_df.empty:
        main_df = df
    else:
        main_df = main_df.join(df)

print(main_df.head())
print(len(main_df))

for c in main_df.columns:
    print(c)


main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)

main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))

print(main_df[[f'{RATIO_TO_PREDICT}_close', 'future', 'target']].head(30))


times = sorted(main_df.index.values)

last_5pct = times[-int(0.05*len(times))]

print(last_5pct)

main_df_validation = main_df[main_df.index >= last_5pct]
main_df = main_df[main_df.index < last_5pct]

def preprocessing_df(df):
    df = df.drop('future', 1)

    for col in df.columns:
        if col != 'target':
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)

    # print(df.head())
    # for c in df.columns:
    #     print(c)


    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            # t1 = i[-1]
            # t2 = np.array(prev_days)
            # t3 = np.array(prev_days, i[-1])
            # t4 = [t2]
            # if np.array_equal(t2,t3):
            #     str = ''
            sequential_data.append([np.array(prev_days), i[-1]])


    random.shuffle(sequential_data)

    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            buys.append([seq, target])
        elif target == 1:
            sells.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells
    random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y

#preprocessing_df(main_df)

X_train, y_train = preprocessing_df(main_df)
X_val, y_val = preprocessing_df(main_df_validation)


print(f"train data: {len(X_train)} validation: {len(X_val)}")
print(f"Dont buys: {y_train.count(0)}, buys: {y_train.count(1)}")
print(f"VALIDATION Dont buys: {y_val.count(0)}, buys: {y_val.count(1)}")



model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1:]), return_sequences=True))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

#history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val), callbacks=[tensorboard, checkpoint])

history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val),callbacks=[tensorboard])
