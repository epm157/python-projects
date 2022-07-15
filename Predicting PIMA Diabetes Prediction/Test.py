import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
import xgboost
import datetime
from sklearn.model_selection import cross_val_score


import tensorflow as tf
print(tf.__version__)

data = pd.read_csv("./data/pima-data.csv")
print(data.shape)
print(data.head(5))


print(data.isnull().values.any())


corMat = data.corr()
top_corr_features = corMat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(data[top_corr_features].corr(), annot=True, cmap='RdYlGn')
plt.show()

print(data.corr())

diabetes_map = {True: 1, False:0}
data['diabetes'] = data['diabetes'].map(diabetes_map)
print(data.head())

diabetes_count_true = len(data.loc[data['diabetes'] == True])
diabetes_count_false = len(data.loc[data['diabetes'] == False])
(diabetes_count_true, diabetes_count_false)

feature_columns = ['num_preg', 'glucose_conc', 'diastolic_bp', 'insulin', 'bmi', 'diab_pred', 'age', 'skin']
predicted_class = ['diabetes']

X = data[feature_columns].values
y = data[predicted_class].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=10)



print("total number of rows : {0}".format(len(data)))
print("total number of rows : {0}".format(len(data)))
print("number of rows missing glucose_conc: {0}".format(len(data.loc[data['glucose_conc'] == 0])))
print("number of rows missing glucose_conc: {0}".format(len(data.loc[data['glucose_conc'] == 0])))
print("number of rows missing diastolic_bp: {0}".format(len(data.loc[data['diastolic_bp'] == 0])))
print("number of rows missing insulin: {0}".format(len(data.loc[data['insulin'] == 0])))
print("number of rows missing bmi: {0}".format(len(data.loc[data['bmi'] == 0])))
print("number of rows missing diab_pred: {0}".format(len(data.loc[data['diab_pred'] == 0])))
print("number of rows missing age: {0}".format(len(data.loc[data['age'] == 0])))
print("number of rows missing skin: {0}".format(len(data.loc[data['skin'] == 0])))


fill_values = Imputer(missing_values=0, strategy='mean', axis=0)
X_train = fill_values.fit_transform(X_train)
X_test = fill_values.fit_transform(X_test)


random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train.ravel())


predict_train_data = random_forest_model.predict(X_test)

print(metrics.accuracy_score(y_test, predict_train_data))


params = {
"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
}


classifier=xgboost.XGBClassifier()
random_search = RandomizedSearchCV(classifier, param_distributions=params)

def timer(star_time=None):
    if not star_time:
        star_time = datetime.now()
        return star_time
    elif star_time:
        thour, temp_sec = divmod((datetime.now - star_time).total_seconds, 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

#start_time = timer(None)
random_search.fit(X, y.ravel())
#timer(start_time)


print(random_search.best_estimator_)


score = cross_val_score(classifier, X, y.ravel(), cv=10)

print(score.mean())






