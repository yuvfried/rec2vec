import pandas as pd
from nltk import RegexpTokenizer, PorterStemmer

raw_techs = pd.read_csv("raw_techs.csv")

corpus = corpus.str.lower()
corpus = corpus.apply(lambda sentence: RegexpTokenizer(r'\w+').tokenize(sentence))


# function to apply each sentence in the corpus
# tokenized_sentence is a list
# techs is a series of stemmed cook-techniques
def stem_techs(tokenized_sentence, techs):
    ps = PorterStemmer()
    output_sentence = copy.deepcopy(tokenized_sentence)
    words_that_stemmed = []
    for i, word in enumerate(tokenized_sentence):
        stemmed = ps.stem(word)
        if (techs == stemmed).any():
            output_sentence[i] = stemmed
            words_that_stemmed.append(stemmed)
    return pd.Series([output_sentence, words_that_stemmed])


"""Cell to apply desired stemmings, approx 15 min"""

# raw_techs['stemmed'] is the stemmed series of raw_techs
df_after_stem = corpus.apply(lambda sentence: stem_techs(sentence, raw_techs['stemmed']))
df_after_stem.columns = ['sentence', 'stemmed']
df_after_stem['sentence'] = df_after_stem['sentence'].apply(lambda x: " ".join(x))
stemmed_counts = pd.Series(list(chain.from_iterable(df_after_stem["stemmed"].tolist()))).value_counts()
