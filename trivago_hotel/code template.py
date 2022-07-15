import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import xgboost
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU,PReLU,ELU
from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend as K


columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']

LotFrontage = 'LotFrontage'
BsmtCond = 'BsmtCond'
BsmtQual = 'BsmtQual'
FireplaceQu = 'FireplaceQu'
GarageType = 'GarageType'
GarageFinish = 'GarageFinish'
GarageQual = 'GarageQual'
GarageCond = 'GarageCond'
MasVnrType = 'MasVnrType'
MasVnrArea = 'MasVnrArea'
BsmtExposure = 'BsmtExposure'
BsmtFinType2 = 'BsmtFinType2'
SalePrice = 'SalePrice'
Id = 'Id'


df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

print(df_train.head())

print(df_train['MSZoning'].value_counts())

sn.heatmap(df_train.isnull(), yticklabels=False, cbar=False)
plt.show()






dataset2 = df_train.copy()
dataset2 = dataset2[['MSSubClass','LotFrontage','LotArea', 'YearBuilt', 'MasVnrArea', 'TotalBsmtSF',
'GrLivArea', 'GarageYrBlt','WoodDeckSF']]

dataset2.dropna()

fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(dataset2.shape[1]):
    plt.subplot(6, 3, i + 1)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i])

    vals = np.size(dataset2.iloc[:, i].unique())
    if vals >= 100:
        vals = 100

    plt.hist(dataset2.iloc[:, i], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


dataset2.corrwith(df_train.SalePrice).plot.bar(
    figsize = (20, 10), title = 'Correlation with E Signed',
    fontsize = 15, rot = 45, grid = True)
plt.show()





## Correlation Matrix
sn.set(style="white")

# Compute the correlation matrix
corr = dataset2.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))

# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()





def preprocess_df(df):

    df[LotFrontage] = df[LotFrontage].fillna(df[LotFrontage].mean())
    df.drop(['Alley'], axis=1, inplace=True)
    df[BsmtCond] = df[BsmtCond].fillna(df[BsmtCond].mode()[0])
    df[BsmtQual] = df[BsmtQual].fillna(df[BsmtQual].mode()[0])
    df[FireplaceQu] = df[FireplaceQu].fillna(df[FireplaceQu].mode()[0])
    df[GarageType] = df[GarageType].fillna(df[GarageType].mode()[0])
    df.drop(['GarageYrBlt'], axis=1, inplace=True)
    df[GarageFinish] = df[GarageFinish].fillna(df[GarageFinish].mode()[0])
    df[GarageQual] = df[GarageQual].fillna(df[GarageQual].mode()[0])
    df[GarageCond] = df[GarageCond].fillna(df[GarageCond].mode()[0])
    df[MasVnrType] = df[MasVnrType].fillna(df[MasVnrType].mode()[0])
    df[MasVnrArea] = df[MasVnrArea].fillna(df[MasVnrArea].mode()[0])
    df[BsmtExposure] = df[BsmtExposure].fillna(df[BsmtExposure].mode()[0])
    df[BsmtFinType2] = df[BsmtFinType2].fillna(df[BsmtFinType2].mode()[0])
    df.drop(['PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
    df.drop(['Id'], axis=1, inplace=True)
    return df






print(len(columns))

df_train = preprocess_df(df_train)
df_test = preprocess_df(df_test)


def category_onehot_multicols(columns, df_categorical):
    df_final = pd.DataFrame()
    for column in columns:
        print(column)
        df_temp = pd.get_dummies(df_categorical[column], drop_first=True)
        df_categorical.drop([column], axis=1, inplace=True)
        df_final = pd.concat([df_final, df_temp], axis = 1)

    df_final = pd.concat([df_categorical, df_final], axis=1)

    return df_final



final_df = pd.concat([df_train, df_test], axis=0)

print(final_df.shape)

final_df = category_onehot_multicols(columns, final_df)

final_df.dropna(inplace=True)

final_df = final_df.loc[:, ~final_df.columns.duplicated()]

print(final_df.shape)
print(final_df.isnull().sum())



sn.heatmap(final_df.isnull(), yticklabels=False,cbar=False,cmap='coolwarm')
plt.show()

split = 1422
df_train = final_df.iloc[:split, :]
df_test = final_df.iloc[split:, :]

df_test.drop([SalePrice], axis=1, inplace=True)

print(df_test.shape)



X_train_all = df_train.drop([SalePrice], axis=1)
y_train_all = df_train[SalePrice]

X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=1)


classifier = xgboost.XGBRegressor()
classifier.fit(X_train, y_train)

y_val_pred = classifier.predict(X_val)

print(y_val.shape)
print(y_val_pred.shape)

print(y_val[10:20])
print(y_val_pred[10:20])



print('Mean Absolute Error:', metrics.mean_absolute_error(y_val, y_val_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_val, y_val_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_val, y_val_pred)))


#df_temp = pd.DataFrame([y_val, y_val_pred], axis=1)



booster=['gbtree','gblinear']
base_score=[0.25,0.5,0.75,1]

n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]

# Define the grid of hyperparameters to search
grid_params = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }

gd_sr = GridSearchCV(estimator=classifier,
            param_grid=grid_params,
            cv=5,
            scoring = 'neg_mean_absolute_error',
            n_jobs = 4,
            verbose = 5,
            return_train_score = True)
#gd_sr.fit(X_train, y_train)


def build_model():
    model = Sequential()
    model.add(Dense(units=256, kernel_initializer='he_uniform', activation='relu', input_dim=175))
    model.add(Dense(units=256, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=256, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, kernel_initializer='he_uniform'))

    def root_mean_squared_error(y_true, y_pred):
        loss = K.square(y_pred - y_true)
        loss = K.mean(loss)
        loss = K.sqrt(loss)
        return loss


    model.compile(loss=root_mean_squared_error, optimizer='Adamax')
    return model

model = build_model()
model_history = model.fit(X_train_all.values, y_train_all.values, validation_split=0.20, batch_size=128, nb_epoch=100000)

ann_pred = model.predict(df_test.drop([SalePrice], axis=1).values)



filename = 'data/finalized_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))

y_pred = classifier.predict(df_test)


pred = pd.DataFrame(y_pred)
sub_df = pd.read_csv('data/sample_submission.csv')
datasets = pd.concat([sub_df[Id], pred], axis=1)
datasets.column = [Id, SalePrice]
datasets.to_csv('data/submition.csv', index=False)









