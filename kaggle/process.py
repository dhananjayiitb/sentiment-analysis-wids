from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
import pandas as pd

training = pd.read_csv("lemmatized_training.csv")
count_vect = CountVectorizer(stop_words=["feel", "and", "be", "to", "the", "that", "of", "my", "not", "feeling"], max_df=0.8)
tfidf_transformer = TfidfTransformer()
train_counts = count_vect.fit_transform(training.text)
train_tfidf = tfidf_transformer.fit_transform(train_counts)

clf = ComplementNB().fit(train_tfidf, training.label)

test = pd.read_csv("lemmatized_test.csv")
test_counts = count_vect.transform(test.text)
test_tfidf = tfidf_transformer.transform(test_counts)

predictions = pd.read_csv("sample.csv")
predictions["label"] = clf.predict(test_tfidf)
predictions.to_csv("submission.csv", index = False)