import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics


df_mails = pd.read_csv('spam.csv', encoding= 'latin-1')
print(df_mails.head())

print(df_mails.isnull().sum())

print(df_mails.v1.value_counts())


#mapping = {'spam': 1, 'ham': 0}
#df_mails.v1 = df_mails.v1.map(mapping)

df_mails.v1.value_counts().plot(kind='bar')
plt.show()


df_spam = df_mails[df_mails.v1 == 'spam']
print(df_spam.head())

nlp = spacy.load('en_core_web_sm')
print('.'.isalpha())

spam_token = []
famous_keyword = set([])
for spam in tqdm(np.array(df_spam.v2)):
    doc = nlp(spam.lower())
    for token in doc:
        if not token.is_stop and token.text.isalpha() and not token.text.isdigit():
            spam_token.append(token.text)
            if token.pos_ == 'NOUN' or token.pos_ == 'PRON' or token.pos_ == 'PROPN':
                famous_keyword |= set([token.text])


famous_keyword = list(famous_keyword)
print(famous_keyword[:10])

print(spam_token[:10])

freq_spam = FreqDist(spam_token)
print(freq_spam)

plt.figure(figsize=(15, 10))
freq_spam.plot(50)
plt.show()


corpus = []
for i in tqdm(range(df_mails.shape[0])):
    msg = df_mails.v2[i]
    msg = re.sub('[^a-zA-Z]', ' ', msg)
    msg = msg.lower()
    doc = nlp(msg)
    tokens_no_stop = [token.lemma_ for token in doc if not token.is_stop and not token.text.isspace()]
    msg = ' '.join(tokens_no_stop)
    corpus.append(msg)


print(corpus[:10])

X = corpus
y = df_mails.v1


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=34)




#vectorizer = TfidfVectorizer()
#X_train_tfidf = vectorizer.fit_transform(X_train)

#classifier = LinearSVC()
#classifier.fit(X_train_tfidf, y_train)



text_classifier = Pipeline([('tfidf', TfidfVectorizer()), ('classifier', LinearSVC())])
text_classifier.fit(X_train, y_train)


prediction = text_classifier.predict(X_test)
print(prediction)

cm = metrics.confusion_matrix(prediction, y_test)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True)
plt.show()

print(metrics.classification_report(y_test, prediction))

print(f'Accuracy: {metrics.accuracy_score(y_test, prediction)}')


sentence_ham = 'Hello Sir. How are you?'
sentence_spam = 'Weekly Lottery Participation. Win upto $10,000.'

print(text_classifier.predict([sentence_ham]))
print(text_classifier.predict([sentence_spam]))








