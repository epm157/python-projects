'''

Handle missing values,
Scale data
drop na
drop duplicates

put city id back in but crop rare values

collaborative filtering for missing value?


Not onlyc click, but also complete reservation, Where the clicks were, which times (season, days, hours),
which os users tend to click more? Desktop or device? Distance of the users to the hotel while booking,
How many days before arriving did they book/click? wh

'''

import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import xgboost
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics, preprocessing

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU,PReLU,ELU
from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend as K


COLUMN_TARGET = 'n_clicks'
COLUMN_ID = 'hotel_id'
COLUMN_CITY_ID = 'city_id'
COLUMN_CONTENT_SCORE = 'content_score'
COLUMN_N_IMAGES = 'n_images'
COLUMN_DISTANCE_TO_CENTER = 'distance_to_center'
COLUMN_STARS = 'stars'
COLUMN_N_REVIEWS = 'n_reviews'
COLUMN_AVG_RATING = 'avg_rating'
COLUMN_AVG_PRICE = 'avg_price'
COLUMN_AVG_RANK = 'avg_rank'
COLUMN_AVG_SAVING_PERCENT = 'avg_saving_percent'
COLUMN_N_CLICKS = 'n_clicks'



dataset_train = pd.read_csv('data/train_set.csv')
dataset_test = pd.read_csv('data/test_set.csv')

print(dataset_train.head())



df_train = dataset_train.copy()
df_test = dataset_test.copy()

dataset_train.set_index(COLUMN_ID, inplace=True)
dataset_test.set_index(COLUMN_ID, inplace=True)

print(dataset_train.isnull().sum())


def plot_missing_values_heatmap(data_frame):
    if data_frame.empty :
        return
    sn.heatmap(data_frame.isnull(), yticklabels=False, cbar=False)
    plt.show()

def plot_correlation_target(data_frame, target_column_name):
    #copy dataframe
    data_frame_without_target = data_frame.copy()
    #drop target column as we do not need for the correlation plot
    data_frame_without_target.drop([target_column_name], axis=1, inplace=True)

    # hotel id is not relevant
    #data_frame.drop([COLUMN_ID], axis=1, inplace=True)
    #data_frame_without_target.drop([COLUMN_ID], axis=1, inplace=True)

    data_frame_without_target.corrwith(data_frame[COLUMN_TARGET]).plot.bar(
        figsize=(20, 10), title='Correlation with n clicks',
        fontsize=15, rot=45, grid=True)
    plt.show()


def plot_correlation_matrix(data_frame):
    ## Correlation Matrix
    sn.set(style="white")

    # Compute the correlation matrix
    corr = data_frame.corr()

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

def plot_histogram(dataset):
    fig = plt.figure(figsize=(15, 12))
    plt.suptitle('Histograms of Numerical Columns', fontsize=20)
    for i in range(dataset.shape[1]):
        plt.subplot(6, 3, i + 1)
        f = plt.gca()
        f.set_title(dataset.columns.values[i])

        vals = np.size(dataset.iloc[:, i].unique())
        if vals >= 100:
            vals = 100

        plt.hist(dataset.iloc[:, i], bins=vals, color='#3F5D7D')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

plot_missing_values_heatmap(dataset_train.copy())
plot_correlation_matrix(dataset_train.copy())
plot_correlation_target(dataset_train.copy(), COLUMN_TARGET)
plot_histogram(dataset_train)
plot_histogram(dataset_test)


def category_onehot(df_categorical, columns):
    df_final = pd.DataFrame()
    for column in columns:
        print(column)
        df_temp = pd.get_dummies(df_categorical[column], drop_first=True)
        df_categorical.drop([column], axis=1, inplace=True)
        df_final = pd.concat([df_final, df_temp], axis=1)

    df_final = pd.concat([df_categorical, df_final], axis=1)

    return df_final

def cleanup_missing_data(dataset_train, dataset_test):
    null_data = dataset_train[dataset_train.isnull().any(axis=1)]
    plot_missing_values_heatmap(null_data)

    print(null_data.describe())
    print(dataset_train.shape)
    print(null_data.shape)

    high_missing_columns = [COLUMN_CITY_ID, COLUMN_CONTENT_SCORE, COLUMN_N_IMAGES, COLUMN_DISTANCE_TO_CENTER,
                            COLUMN_STARS, COLUMN_N_REVIEWS]
    # train data
    dataset_missed_value_columns = dataset_train[high_missing_columns]
    null_data_train = dataset_missed_value_columns[dataset_missed_value_columns.isnull().any(axis=1)]
    plot_missing_values_heatmap(null_data_train)
    print(null_data_train.shape)

    # test data
    dataset_missed_value_columns = dataset_test[high_missing_columns]
    null_data = dataset_missed_value_columns[dataset_missed_value_columns.isnull().any(axis=1)]
    plot_missing_values_heatmap(null_data)
    print(null_data.shape)

    temp = null_data_train.index.values.tolist()
    dataset_train.drop(temp, inplace=True)

    mean_avg_rating_train_dataset = dataset_train[COLUMN_AVG_RATING].mean()
    mean_avg_price_train_dataset = dataset_train[COLUMN_AVG_PRICE].mean()
    mean_avg_saving_percentage_train_dataset = dataset_train[COLUMN_AVG_SAVING_PERCENT].mean()
    mean_content_score_train_dataset = dataset_train[COLUMN_CONTENT_SCORE].mean()
    mean_distance_to_center_train_dataset = dataset_train[COLUMN_DISTANCE_TO_CENTER].mean()
    mean_n_images_train_dataset = dataset_train[COLUMN_N_IMAGES].mean()
    mean_n_reviews_train_dataset = dataset_train[COLUMN_N_REVIEWS].mean()
    mean_stars_train_dataset = dataset_train[COLUMN_STARS].mean()


    dataset_train[COLUMN_AVG_RATING] = dataset_train[COLUMN_AVG_RATING].fillna(mean_avg_rating_train_dataset)
    dataset_train[COLUMN_AVG_PRICE] = dataset_train[COLUMN_AVG_PRICE].fillna(mean_avg_price_train_dataset)
    dataset_train[COLUMN_AVG_SAVING_PERCENT] = dataset_train[COLUMN_AVG_SAVING_PERCENT].fillna(mean_avg_saving_percentage_train_dataset)

    dataset_test[COLUMN_AVG_RATING] = dataset_test[COLUMN_AVG_RATING].fillna(mean_avg_rating_train_dataset)
    dataset_test[COLUMN_AVG_PRICE] = dataset_test[COLUMN_AVG_PRICE].fillna(mean_avg_price_train_dataset)
    dataset_test[COLUMN_AVG_SAVING_PERCENT] = dataset_test[COLUMN_AVG_SAVING_PERCENT].fillna(mean_avg_saving_percentage_train_dataset)

    dataset_test[COLUMN_CONTENT_SCORE] = dataset_test[COLUMN_CONTENT_SCORE].fillna(mean_content_score_train_dataset)
    dataset_test[COLUMN_DISTANCE_TO_CENTER] = dataset_test[COLUMN_DISTANCE_TO_CENTER].fillna(mean_distance_to_center_train_dataset)
    dataset_test[COLUMN_N_IMAGES] = dataset_test[COLUMN_N_IMAGES].fillna(mean_n_images_train_dataset)
    dataset_test[COLUMN_N_REVIEWS] = dataset_test[COLUMN_N_REVIEWS].fillna(mean_n_reviews_train_dataset)
    dataset_test[COLUMN_STARS] = dataset_test[COLUMN_STARS].fillna(mean_stars_train_dataset)

    null_data = dataset_train[dataset_train.isnull().any(axis=1)]
    plot_missing_values_heatmap(null_data)

    print(null_data.describe())
    print(dataset_train.shape)
    print(null_data.shape)
    print(dataset_train.isnull().sum())

    print(dataset_test.shape)
    print(dataset_test.isnull().sum())

    return dataset_train, dataset_test


dataset_train, dataset_test = cleanup_missing_data(dataset_train, dataset_test)
dataset_train.drop([COLUMN_CITY_ID], axis=1, inplace=True)
dataset_test.drop([COLUMN_CITY_ID], axis=1, inplace=True)

dataset = dataset_train.append(dataset_test)
plot_missing_values_heatmap(dataset.copy())
plot_correlation_matrix(dataset.copy())
plot_correlation_target(dataset.copy(), COLUMN_TARGET)
plot_histogram(dataset_train)
plot_histogram(dataset_test)



#dataset = category_onehot(dataset, [COLUMN_CITY_ID])
print(dataset.shape)
#print(dataset_train[COLUMN_CITY_ID].nunique())
#initial train+test 528649


print(dataset_train[COLUMN_N_CLICKS].nunique())

print(dataset_train.groupby(COLUMN_N_CLICKS).count())

print(dataset.isnull().sum())


split = 395925


df_train = dataset.iloc[:split, :]
df_test = dataset.iloc[split:, :]


max_avg_price = df_train[COLUMN_AVG_PRICE].max()
max_avg_rank = df_train[COLUMN_AVG_RANK].max()
max_avg_rating = df_train[COLUMN_AVG_RATING].max()
max_avg_saving_percentage = df_train[COLUMN_AVG_SAVING_PERCENT].max()
max_content_score = df_train[COLUMN_CONTENT_SCORE].max()
max_distance_to_center = df_train[COLUMN_DISTANCE_TO_CENTER].max()
max_n_images = df_train[COLUMN_N_IMAGES].max()
max_n_reviews = df_train[COLUMN_N_REVIEWS].max()
max_star = df_train[COLUMN_STARS].max()
max_n_clicks = df_train[COLUMN_N_CLICKS].max()


'''
df_train[COLUMN_AVG_PRICE] = df_train[COLUMN_AVG_PRICE] / max_avg_price
df_train[COLUMN_AVG_RANK]  = df_train[COLUMN_AVG_RANK] / max_avg_rank
df_train[COLUMN_AVG_RATING] = df_train[COLUMN_AVG_RATING] / max_avg_rating
df_train[COLUMN_AVG_SAVING_PERCENT] = df_train[COLUMN_AVG_SAVING_PERCENT] / max_avg_saving_percentage
df_train[COLUMN_CONTENT_SCORE] = df_train[COLUMN_CONTENT_SCORE] / max_content_score
df_train[COLUMN_DISTANCE_TO_CENTER] = df_train[COLUMN_DISTANCE_TO_CENTER] / max_distance_to_center
df_train[COLUMN_N_IMAGES] = df_train[COLUMN_N_IMAGES] / max_n_images
df_train[COLUMN_N_REVIEWS] = df_train[COLUMN_N_REVIEWS] / max_n_reviews
df_train[COLUMN_STARS] = df_train[COLUMN_STARS] / max_star
df_train[COLUMN_N_CLICKS] = df_train[COLUMN_N_CLICKS] / max_n_clicks






x = df_train.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_train = pd.DataFrame(x_scaled)
'''
print(df_train.shape)


'''



df_test[COLUMN_AVG_PRICE] = df_test[COLUMN_AVG_PRICE] / max_avg_price
df_test[COLUMN_AVG_RANK]  = df_test[COLUMN_AVG_RANK] / max_avg_rank
df_test[COLUMN_AVG_RATING] = df_test[COLUMN_AVG_RATING] / max_avg_rating
df_test[COLUMN_AVG_SAVING_PERCENT] = df_test[COLUMN_AVG_SAVING_PERCENT] / max_avg_saving_percentage
df_test[COLUMN_CONTENT_SCORE] = df_test[COLUMN_CONTENT_SCORE] / max_content_score
df_test[COLUMN_DISTANCE_TO_CENTER] = df_test[COLUMN_DISTANCE_TO_CENTER] / max_distance_to_center
df_test[COLUMN_N_IMAGES] = df_test[COLUMN_N_IMAGES] / max_n_images
df_test[COLUMN_N_REVIEWS] = df_test[COLUMN_N_REVIEWS] / max_n_reviews
df_test[COLUMN_STARS] = df_test[COLUMN_STARS] / max_star
df_test[COLUMN_N_CLICKS] = df_test[COLUMN_N_CLICKS] / max_n_clicks

'''

X_train_all = df_train.drop([COLUMN_N_CLICKS], axis=1)
y_train_all = df_train[COLUMN_N_CLICKS]

X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=1)


def build_model():
    model = Sequential()
    model.add(Dense(units=128, kernel_initializer='he_uniform', activation='relu', input_dim=9))
    model.add(Dropout(0.5))
    model.add(Dense(units=128, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=128, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=128, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, kernel_initializer='he_uniform'))

    def custom_error(y_true, y_pred):
        loss = K.square(y_pred - y_true)
        loss = loss * (K.log(y_true + 1) + 1)
        loss = K.mean(loss)
        #loss = K.sqrt(loss)
        return loss


    model.compile(loss=custom_error, optimizer='Adamax')
    return model


model = build_model()
model_history = model.fit(X_train_all.values, y_train_all.values, validation_split=0.20, batch_size=2048, nb_epoch=500)




ann_pred = model.predict(df_test.drop([COLUMN_N_CLICKS], axis=1).values)

ann_pred = ann_pred * max_n_clicks




plt.plot(model_history.history['loss'][200:])
plt.plot(model_history.history['val_loss'][200:])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()



history_dict = model_history.history
print(history_dict.keys())

plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()



