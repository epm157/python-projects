import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


human_data = pd.read_csv('human_data.txt', sep='\t')
print(human_data.head())

def get_kmers(sequence, size=6):
    result = [sequence[x: x+size].lower() for x in range(len(sequence) - size+1)]
    return result

human_data['words'] = human_data.apply(lambda x: get_kmers(x['sequence']), axis=1)
human_data = human_data.drop('sequence', axis=1)
print(human_data.head())

human_texts = list(human_data['words'])
for item in range(len(human_texts)):
    human_texts[item] = ' '.join(human_texts[item])

print(human_texts[2])

y_data = human_data.iloc[:, 0].values

cv = CountVectorizer(ngram_range=(4, 4))
X = cv.fit_transform(human_texts)
print(X.shape)


human_data['class'].value_counts().sort_index().plot.bar()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y_data,
                                                    test_size = 0.20,
                                                    random_state=42)

classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))












