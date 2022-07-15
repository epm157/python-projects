import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering


dataset = pd.read_csv('shopping_data.csv')
X = dataset.iloc[:, 3:5].values

dendogram = shc.linkage(X, method = 'ward')

plt.figure(figsize = (10, 7))
plt.title('Customer Dendograms')
dend = shc.dendrogram(dendogram)
plt.show()

classifier = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
classifier.fit_predict(X)

plt.figure(figsize = (10, 7))
plt.scatter(X[:, 0], X[:, 1], c = classifier.labels_, cmap = 'rainbow')
plt.show()
