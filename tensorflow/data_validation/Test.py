import pandas as pd
import numpy as np

import tensorflow as tf
print(tf.__version__)

import tensorflow_data_validation as tfdv



dataset = pd.read_csv('pollution-small.csv')
print(dataset.shape)
print(dataset.describe())

training_data = dataset[:1600]
print(training_data.shape)
print(training_data.describe())

test_data = dataset[1600:]
print(test_data.shape)
print(test_data.describe())


train_stats = tfdv.generate_statistics_from_dataframe(training_data)





