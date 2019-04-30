import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from textblob import Word

train_df = pd.read_csv('poems.csv')
train_df.head()

#converting all the character in to lower case
train_df['content'] = train_df['content'].apply(lambda x: " ".join(x.lower() for x in x.split()))
train_df['content'].head()

#removing punctuation from the catagorical data
train_df['content'] = train_df['content'].str.replace('[^\w\s]','')
train_df['content'].head()

import nltk
nltk.download('stopwords')

#Removing stopwords from the categorical data
stop = stopwords.words('english')
train_df['content'] = train_df['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
train_df['content'].head()

#stemming the words so all forms of word will have same meaning
from nltk.stem import PorterStemmer
st = PorterStemmer()
train_df['content'] = train_df['content'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
train_df['content'].head()

#indexing of topic feature
train_df['TopicIndex'] = pd.factorize(train_df['type'], sort = True)[0] + 1

train_df.head(50)

from sklearn_pandas import DataFrameMapper
from sklearn.feature_extraction.text import TfidfVectorizer

train_df = train_df.fillna('')

#converting title feature in to the vector form using TfidVectorizer so we can input this feature in to our model
mapper_title = DataFrameMapper([
    ('content', TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
 stop_words= 'english',ngram_range=(1,1))),
    
], default = False)

features_title = mapper_title.fit_transform(train_df)
labels_title = train_df['type']

# Apply Train-Test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(features_title, labels_title, test_size=0.20, random_state=42)


# MODEL SVM
from sklearn import svm 
model = svm.SVC(kernel='linear') 
model.fit(x_train, y_train) 
prediction = model.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, prediction)
print("SVm model accuracy(in %):",accuracy*100)


#MODEL NAIVE BAYES
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.fit(x_train, y_train) 
y_pred = gnb.predict(x_test)
print("Gaussian Naive Bayes model accuracy(in %):",accuracy_score(y_test, y_pred)*100)
