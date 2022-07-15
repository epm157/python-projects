import pandas as pd
import numpy as np
import datetime


'''
One Hot Encoding with many labels
'''
df = pd.read_csv('data/mercedesbenz.csv', usecols=['X1', 'X2'])
print(df.head())

for col in df:
    print('Col: ', df[col].name , df[col].unique())

print(pd.get_dummies(df, drop_first=True).shape)

print(df.X2.value_counts().sort_values(ascending=False).head(20))

top10_labels = [y for y in df.X2.value_counts().sort_values(ascending=False).head(10).index]
print(top10_labels)

def one_hot_encoding_top_x(df, variable, top_x_labels):
    for label in top_x_labels:
        df[variable+'_'+label] = np.where(df[variable] == label, 1, 0)

one_hot_encoding_top_x(df, 'X2', top10_labels)
print(df.head(10))


'''
One Hot Encoding with Count Frequency encoding
'''
df = pd.read_csv('data/mercedesbenz.csv', usecols=['X1', 'X2'])
print(df.head())

df_freq_map = df.X2.value_counts().to_dict()
print(df_freq_map)


df.X2 = df.X2.map(df_freq_map)

print(df.head())



'''
Ordinal Encoding
'''
df_base = datetime.datetime.today()
df_date_list = [df_base - datetime.timedelta(days=x) for x in range(20)]
df = pd.DataFrame(df_date_list)
df.columns = ['day']
print(df)



df['day_of_week'] = df['day'].dt.weekday_name
print(df.head())

weekday_map = {'Monday':1,
               'Tuesday':2,
               'Wednesday':3,
               'Thursday':4,
               'Friday':5,
               'Saturday':6,
               'Sunday':7
}

df['day_ordinal'] = df['day_of_week'].map(weekday_map)
print(df.head(10))



