from nltk.tokenize import word_tokenize as wt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

df = pd.read_csv("test.csv")
lt = WordNetLemmatizer()
wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}
expansion = pd.read_csv("expansion.csv")

for i in range(len(df['text'])):
    lemmatized = ""
    pos_tags = pos_tag(df['text'][i].split())
    for e in pos_tags:
        new_word = lt.lemmatize(e[0], pos = wordnet_map.get(e[1][0], wordnet.NOUN))
        for j in range(len(expansion['word'])):
            if new_word == expansion['word'][j]: 
                new_word = expansion['expanded'][j]
                break
        if(len(new_word) == 1): 
            continue
        lemmatized += new_word + " "
    df['text'][i] = lemmatized[:-1]

df.to_csv("lemmatized_test.csv", index=False)