import pandas as pd
from dateutil import parser
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn


dataset = pd.read_csv('/Users/ehsan/Dropbox/junk/ML/Machine Learning Practical 6 Real-World Applications/4. Directing Customers to Subscription Through App Behavior Analysis/appdata10.csv')

print(dataset.head(10))
print(dataset.describe())

dataset['hour'] = dataset.hour.str.slice(1, 3).astype(int)

dataset2 = dataset.copy().drop(columns = ['user', 'screen_list', 'enrolled_date',
                                           'first_open', 'enrolled'])

plt.suptitle('Histograms of Numerical Columns', fontsize = 2)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(3, 3, i)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i - 1])

    vals = np.size(dataset2.iloc[:, i - 1].unique())

    plt.hist(dataset2.iloc[:, i - 1], bins = vals, color = '#3F5D7D')
plt.tight_layout(rect = [0, 0.3, 1, 0.95])
plt.show()



dataset2.corrwith(dataset.enrolled).plot.bar(figsize = (20, 10),
                                             title = 'Correlation with Reposnse variable',
                                             fontsize = 15,
                                             rot = 45,
                                           grid = True)
plt.show()





corr = dataset2.corr()

sn.set(style="white", font_scale=2)

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))
f.suptitle("Correlation Matrix", fontsize = 40)

# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()








print(dataset.dtypes)
dataset['first_open'] = [parser.parse(row_date) for row_date in dataset['first_open']]
dataset['enrolled_date'] = [parser.parse(date_row) if isinstance(date_row, str) else date_row for date_row in dataset['enrolled_date']]

dataset['difference'] = (dataset.enrolled_date - dataset.first_open).astype('timedelta64[h]')


plt.hist(dataset['difference'].dropna(), color = '#3F5D7D')
plt.title('Distribution of Time-Since-Screen-Reached')
plt.show()




plt.hist(dataset['difference'].dropna(), color = '#3F5D7D', range= [0, 100])
plt.title('Distribution of Time-Since-Screen-Reached')
plt.show()



dataset.loc[dataset.difference > 48, 'enrolled'] = 0
dataset = dataset.drop(columns = ['enrolled_date', 'difference', 'first_open'])


top_screens = pd.read_csv('top_screens.csv').top_screens.values
dataset['screen_list'] = dataset.screen_list.astype(str) + ','

for sc in top_screens:
    dataset[sc] = dataset.screen_list.str.contains(sc).astype(int)
    dataset['screen_list'] = dataset.screen_list.str.replace(sc + ',', '')

dataset['Other'] = dataset.screen_list.str.count(',')
dataset = dataset.drop(columns = ['screen_list'])

savings_screens = ["Saving1",
                    "Saving2",
                    "Saving2Amount",
                    "Saving4",
                    "Saving5",
                    "Saving6",
                    "Saving7",
                    "Saving8",
                    "Saving9",
                    "Saving10"]
dataset['SavingCount'] = dataset[savings_screens].sum(axis = 1)
dataset = dataset.drop(columns = savings_screens)

cm_screens = ["Credit1",
               "Credit2",
               "Credit3",
               "Credit3Container",
               "Credit3Dashboard"]
dataset["CMCount"] = dataset[cm_screens].sum(axis = 1)
dataset = dataset.drop(columns = cm_screens)

cc_screens = ["CC1",
                "CC1Category",
                "CC3"]
dataset["CCCount"] = dataset[cc_screens].sum(axis=1)
dataset = dataset.drop(columns=cc_screens)

loan_screens = ["Loan",
               "Loan2",
               "Loan3",
               "Loan4"]
dataset["LoansCount"] = dataset[loan_screens].sum(axis=1)
dataset = dataset.drop(columns=loan_screens)

print(dataset.head())
print(dataset.describe())
print(dataset.columns)
dataset.to_csv('test_appdata10.csv', index = False)
