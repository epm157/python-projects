import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

df = pd.read_csv('Data.csv', encoding = "ISO-8859-1")
print(df.head())

train = df[df['Date'] < '20150101']
test = df[df['Date'] >= '20150101']

data = train.iloc[:, 2:27]
data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)
print(data.head())

list1 = [i for i in range(25)]
new_index = [str(i) for i in list1]
data.columns = new_index
print(data.head())


for index in new_index:
    data[index] = data[index].str.lower()

print(data.head(2))

temp = ' '.join(str(x) for x in data.iloc[1, 0:25])
print(temp)

headlines = []
for row in range(len(data.index)):
    headl = ' '.join(str(x) for x in data.iloc[row, 0:25])
    headlines.append(headl)

print(headlines[2])

countVector = CountVectorizer(ngram_range=(2, 2))
train_dataset = countVector.fit_transform(headlines)

classifier = RandomForestClassifier(n_estimators=200, criterion='entropy')
classifier.fit(train_dataset, train['Label'])


test_transform = []
for row in range(len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row, 2:27]))

test_dataset = countVector.transform(test_transform)
predictions = classifier.predict(test_dataset)

cm = confusion_matrix(test['Label'], predictions)
print(cm)

score = accuracy_score(test['Label'], predictions)
print(score)

report = classification_report(test['Label'], predictions)
print(report)



