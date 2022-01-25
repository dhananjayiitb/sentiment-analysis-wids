from tensorflow.keras.layers import Embedding
import tensorflow as tf
import pandas as pd

data = pd.read_csv("lemmatized_training.csv")

embedding_layer = Embedding(1000, 64)

texts = data["text"]
labels = data["label"]

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(texts)

word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(texts)
data = tf.keras.preprocessing.sequence.pad_sequences(sequences)
