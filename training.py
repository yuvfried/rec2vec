import pandas as pd
from gensim.models import Word2Vec

corpus = pd.read_csv("''''path-of-courpus''''", index_col=0, squeeze=True)

sentences = corpus.tolist()
# make sure that sentences is now a list: each of its elements is list which represent a sentence

model = Word2Vec(sentences,min_count=2, window=5, size=100)

model.save("food2vec_ver1.model")
