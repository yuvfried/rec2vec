import pandas as pd
import copy
import re
from nltk.stem import PorterStemmer
import json
from nltk.tokenize import RegexpTokenizer
from itertools import chain

print("1) load instructions databases expected duration: 00:00")
# load instructions databases
rec_db = pd.read_json("raw_data/full_format_recipes.json")  # load Kaggle Recipe DataBase
instructs = rec_db.directions.dropna()  # retrieve recipe instructions
instructs_flat = list(chain.from_iterable(instructs.tolist()))  # flatten the data
instructs_str = ''.join(instructs_flat)  # instructions: list to string

print("2) split the instruction strings into sentences expected duration: 00:04")
# split the instruction strings into sentences (i.e. according to a "." that comes after a word)
instructs_sentences = re.split("((?<=[A-Za-z])+\.)", instructs_str)
corpus = pd.Series(instructs_sentences)  # store the sentences in a Pandas Series
corpus = corpus[corpus != "."]
corpus = corpus.str.lower()
corpus = corpus.apply(lambda sentence: RegexpTokenizer(r'\w+').tokenize(sentence))

print("3) create a new Series of STEMMED cooking techniques expected duration: 00:00")
# create a new Series of STEMMED cooking techniques
cook_techs = pd.read_csv("raw_data/raw_techs.csv")  # cooking techniques Series
ps = PorterStemmer()
cook_techs["stemmed"] = cook_techs["tech"].apply(lambda x: ps.stem(x))  # keep the stemmed version


# the function stems the cooking techniques that appear in the sentence and in the cooking technique list
def stem_techs(tokenized_sentence, cook_techs_ser):
    """
    :type tokenized_sentence: a sentence tokenized into a list of words
    :type cook_techs_ser: a Series of stemmed cooking techniques
    :return a data frame of sentence and words that have been stemmed
    """
    pstem = PorterStemmer()  # stemmer
    output_sentence = copy.deepcopy(tokenized_sentence)
    stemmed_words = []
    for i, word in enumerate(tokenized_sentence):
        stemmed = pstem.stem(word)
        if (cook_techs_ser == stemmed).any():
            output_sentence[i] = stemmed
            stemmed_words.append(stemmed)
    return pd.Series([output_sentence, stemmed_words])


print("4) stemming cooking techniques of the entire corpus expected duration: 12:10")
""" stemming cooking techniques of the entire corpus (approx 15 min) """
# corpus = corpus.apply(lambda sentence: stem_techs(sentence, cook_techs["stemmed"]))
corpus.columns = ["sentence", "stemmed"]  # rename series

print("5) rejoin each sentence into a single string and save stemmed corpus to file expected duration: 00:02")
# rejoin each sentence into a single string and save stemmed corpus to file
corpus["sentence"] = corpus["sentence"].apply(lambda x: " ".join(x))
corpus.to_csv(r'processed_data/corpus_stemmed.csv', index=None, header=True)  # TODO for debugging purposes

print("6) create and save cooking technique histogram to file expected duration: 00:01")
# create and save cooking technique histogram to file
cook_techs_hist = pd.Series(list(chain.from_iterable(corpus["stemmed"].tolist()))).value_counts()
cook_techs_hist.to_csv(r'processed_data/cook_tech_hist.csv', index=None, header=True)

''' Pre-Precessing status: (1) instruction separated into sentences (2) cooking techniques were stemmed'''

# read Kaggle's ingredients database (train and test)
with open("raw_data/train_ingredients.json", "r") as read_file:
    data_train = json.load(read_file)
train_ingredients = pd.read_json("train_ingredients.json")

with open("raw_data/test_ingredients.json", "r") as read_file:
    data_test = json.load(read_file)
test_ingredients = pd.read_json("test_ingredients.json")

print("7) filter the ingredients from the DB expected duration: 00:00")
# filter the ingredients from the DB
train_ingredients = train_ingredients["ingredients"]
test_ingredients = test_ingredients["ingredients"]
ingredients = train_ingredients.append(test_ingredients)  # merge ingredients databases
ingredients = pd.Series(list(chain.from_iterable(ingredients.tolist())))  # convert to pandas Series

# remove duplicates
ingredients = pd.Series(ingredients.unique())

print("8) expected duration: 00:00")
# for each more-than-one-word ingredient consider both '-' and ' ' delimiter (e.g. 'black olives' & 'black-olives')
dash_delimiter = ingredients.str.replace(" ", "-")
space_delimiter = ingredients.str.replace("-", " ")
ingredients = ingredients.append(dash_delimiter).append(space_delimiter)

print("9) expected duration: 00:00")
# convert ingredient names from plural to singular (e.g. potatoes-->potato)
''' we are aware of other forms of singular origins (https://www.grammarly.com/blog/plural-nouns/), however, these rules satisfied our requirements'''
singular_form1 = pd.Series(ingredients.str.replace("s{1}$", ""))
singular_form2 = pd.Series(ingredients.str.replace("es{1}$", ""))
ingredients = ingredients.append(singular_form1).append(singular_form2)

# remove duplicates
ingredients = pd.Series(ingredients.unique())

# keep only the instructions from the corpus
corpus = corpus["sentence"]

print("10) expected duration: 49:00")
# 1) keep only ingredients that are in the intersection of the ingredients database and the corpus
# 2) keep only the ingredients 'dash' delimiter version (e.g. keep 'black olives' and remove 'black-olives')
for ing in ingredients:
    # remove if not in the corpus more-than-one-word ingredients
    if not corpus.str.contains(ing).any():
        ingredients = ingredients.drop(ingredients[ingredients == ing].index)

    # unify all more-than-one-word ingredients to be separated by a '-'
    elif bool(re.match(".* .*", ing)):
        dashed = re.sub(" ", "-", ing)
        ingredients = ingredients.str.replace(ing, dashed)  # TODO creates duplicates?
        corpus = corpus.str.replace(ing, dashed)

print("11) expected duration: 00:00")
# remove duplicates
ingredients = pd.Series(ingredients.unique())

# save ingredients list and corpus to files
ingredients.to_csv("processed_data/process_ingredients.csv", index=None, header=True)
corpus.to_csv("processed_data/processed_corpus.csv", index=None, header=True)
