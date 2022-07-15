import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pandas_profiling
from tqdm import tqdm
import time

print('Start reading data...')
df = pd.read_csv('mercedesbenz.csv')
print('Dataset read!')
profile = df.profile_report(title='Pandas Profiling Report')

profile.to_file(output_file="output_mercedesbenz.html")
print('Profile created!')

TIP = 'tip'
TOTAL_BILL = 'total_bill'
SEX = 'sex'
SMOKER = 'smoker'
DAY = 'day'

df = sns.load_dataset('tips')
print(df.head())

print(df.corr())

sns.heatmap(df.corr())
plt.show()

sns.jointplot(x=TIP, y=TOTAL_BILL, data=df, kind='hex')
plt.show()

sns.jointplot(x=TIP, y=TOTAL_BILL, data=df, kind='reg')
plt.show()


sns.pairplot(df)
plt.show()

sns.pairplot(df, hue=SEX)
plt.show()

sns.distplot(df[TIP])
plt.show()


sns.distplot(df[TIP], kde=False, bins=10)
plt.show()


sns.distplot(df[TOTAL_BILL])
plt.show()


sns.countplot(SEX, data=df)
plt.show()

sns.countplot(y=SEX, data=df)
plt.show()


sns.barplot(x=TOTAL_BILL, y=SEX, data=df)
plt.show()

sns.barplot(y=TOTAL_BILL, x=SEX, data=df)
plt.show()


sns.boxplot(y=TOTAL_BILL, data=df)
plt.show()

sns.boxplot(SMOKER, TOTAL_BILL, data=df)
plt.show()

sns.barplot(x=TOTAL_BILL, y=SEX, data=df)
plt.show()

sns.boxplot(x=DAY, y=TOTAL_BILL, data=df, palette='rainbow')
plt.show()


sns.boxplot(data=df, orient='v')
plt.show()

sns.boxplot(x=TOTAL_BILL, y=DAY, hue=SMOKER, data=df)
plt.show()

sns.violinplot(x=TOTAL_BILL, y=DAY, data=df, palette='rainbow')
plt.show()



for i in tqdm(range(10000)):
    time.sleep(.01)
    if i > 0 and i % 10 == 0:
        print(i)
