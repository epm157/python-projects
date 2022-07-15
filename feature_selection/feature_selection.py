import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn as sns

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')
X = data.iloc[:,:-1]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range

## Univariate Selection
best_features = SelectKBest(score_func=chi2, k=10)
fit = best_features.fit(X, y)

df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)


print(df_scores)
print(df_columns)


feature_scores = pd.concat([df_columns, df_scores], axis=1)
feature_scores.columns = ['Specs', 'Score']

print(feature_scores)


print(feature_scores.nlargest(10, 'Score'))



## Feature Importance

model = ExtraTreesClassifier()
model.fit(X, y)
print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


## Correlation Matrix with Heatmap

corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
sns.heatmap(data[top_corr_features].corr(), annot=True, cmap='RdYlGn')
plt.show()





